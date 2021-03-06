#!/usr/bin/env python3
"""
Report class for generating all reports.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
import xlsxwriter
import pdfkit

import pandas as pd
import numpy as np

from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

from sec_edgar_extractor.extract import Extractor
from sec_edgar_extractor.documentation import Documentation

#built-in
from asyncio.log import logger
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import namedtuple
import ast
from enum import unique

#my libs
from utils import (
    remove_list_dups,
)
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    LIST_ALL_ACCOUNTS,
    LIST_ALL_FIRMS
)


ex = Extractor(save_intermediate_files=True)
doc = Documentation()
doc.load_documents()

df_config = ex.config.get(mode='df')
xbrl_labels = set( df_config['xbrl'].tolist() )
doc_records = doc.get_records(xbrl_labels)




class Report:
    """Report class to output various types from database."""

    def __init__(self, db, output_path):
        self.db = db
        valid_path = Path(output_path)
        if valid_path.is_dir():
            self.output_path = valid_path
        else:
            print(f'Dir does not exist: {output_path}')
            raise IOError


    def create_report(self, type):
        """Primary entry to create new report"""
        dir_path = Path(self.output_path).parent
        match type:
            case 'long': 
                result = self.report_long(self.output_path)
            case 'accounting_policy':
                df_long = self.report_long()
                df_valid = self.validate(df_long, self.output_path) 
                list_of_df_tables = self.report_accounting_policy_preparation(df_long, df_valid)
                result = self.report_accounting_policy_completion(list_of_df_tables)
            case 'trend':
                df_long = self.report_long()
                #result = template(df_long, dir_path)
                df_valid = self.validate(df_long, self.output_path) 
                df_ACL = self.report_accounting_policy_preparation(df_long, df_valid)
                result = self.trend(df_ACL, dir_path)
            case 'validate':
                df_long = self.report_long()
                df_valid = self.validate(df_long, self.output_path)
                result = True if df_valid.shape[0] > 0 else False

        return result


    def report_long(self, output_path=None):
        """TODO: move to db query.  It is necessary for everything, so just make it automatic."""
        report_name = 'long_output.csv'
        df = self.db.query_database(table_name='filings')
        df['dt_filed'] = pd.to_datetime(df.filed)
        form_categories = ['10-Q', '10-K', '8-K/EX-99.3', '8-K/EX-99.2', '8-K/EX-99.1']         #NOTE: precedence used for sorting and selecting data
        df['form'] = pd.Categorical(df['form'], 
                                    categories=form_categories, 
                                    ordered=True)
        df.sort_values(by=['cik','yr_qtr','form'], inplace=True, ascending=False)
        if df.shape[0] > 0 and output_path:
            report_path = output_path / report_name
            df.to_csv(report_path, index=False)
            logger.info(f'Report saved to path: {output_path}')
        else:
            logger.info(f'No data available to report')
        return df


    def report_accounting_policy_preparation(self, df_long, df_valid):
        """Report preparation for accounting policy by returning a list of 
        dataframes, one for each metric.
        """

        # prepare dataframe
        ciks = df_long.cik.unique().tolist()
        df_tmp = df_long[(pd.isna(df_long['ACL'])==False)]
        df_tmp['ACL_num'] = df_tmp['ACL'].apply(lambda x: np.abs( pd.to_numeric(x, errors='coerce')) )
        df_tmp['Loans_num'] = df_tmp['Loans'].apply(lambda x: np.abs( pd.to_numeric(x, errors='coerce')) )
        
        now = pd.Timestamp(datetime.now())
        initial_qtr = f'{now.year}-{now.quarter}'
        #df_tmp['yr_qtr'] = pd.DatetimeIndex(df_tmp['dt_filed']).year.astype(str) + '-Q' + pd.DatetimeIndex(df_tmp['dt_filed']).quarter.astype(str)
        df_tmp.sort_values(by='dt_filed', inplace=True)
        
        dfcnt = pd.DataFrame( df_tmp['yr_qtr'].value_counts() )
        dfcnt['date'] = pd.PeriodIndex(dfcnt.index, freq='Q').to_timestamp()
        dfcnt['qtrs'] = dfcnt.index
        dfcnt.sort_values(by='date', ascending=False, inplace=True)
        qtrs = dfcnt.qtrs.tolist()

        # get df for each qtr for each account topic (VERY IMPORTANT)
        df_result = pd.DataFrame()
        df_result['cik'] = df_long['cik'].unique()
        df_result.set_index('cik', inplace=True)

        dfsACL = {}
        for idx,qtr in enumerate(qtrs):
            df_tmp1 = df_tmp[df_tmp['yr_qtr'] == qtr]
            df_tmp2 = df_tmp1.sort_values(by=['cik','form'], ascending=True).dropna(subset=['ACL']).drop_duplicates(subset='cik')
            df_tmp3 = df_tmp2.groupby('cik').head(1)
            df_tmp3.set_index('cik', inplace=True)
            if df_tmp3.shape[0] > 0:
                dfsACL[qtr] = df_tmp3

        dfsLoans = {}
        for idx,qtr in enumerate(qtrs):
            df_tmp1 = df_tmp[df_tmp['yr_qtr'] == qtr]
            df_tmp2 = df_tmp1.sort_values(by=['cik','form'], ascending=True).dropna(subset=['Loans']).drop_duplicates(subset='cik')
            df_tmp3 = df_tmp2.groupby('cik').head(1)
            df_tmp3.set_index('cik', inplace=True)
            if df_tmp3.shape[0] > 0:
                dfsLoans[qtr] = df_tmp3

        # create df for metadata and each metric by adding appropriate columns for each qtr
        meta = namedtuple('meta_record', ['cik', 'accn', 'form', 'titles', 'confidence'])
        df_Meta = df_result.copy(deep=True)
        df_ACL = df_result.copy(deep=True)
        df_Loans = df_result.copy(deep=True)
        df_Ratio = df_result.copy(deep=True)
        for key in dfsACL.keys():
            if dfsACL[key].shape[0] > 0:

                col = 'meta'+'|'+key
                df_Meta[col] = None
                dfsACL[key]['cik'] = dfsACL[key].index
                for row in dfsACL[key].to_dict('records'):

                    accn_col = 'accn_10q' if row['form'] in ['10-K','10-Q'] else 'accn_8k'
                    records = df_valid[(df_valid['yr_qtr'] == key) & (df_valid[accn_col] == row['accn'])].to_dict('records')
                    rec_valid = records[0] if len(records) > 0 else {'conf_ACL': 0.0, 'conf_Loans': 0.0}
                    rec = meta(
                        cik = str(row['cik']),
                        accn = row['accn'],
                        form = row['form'],
                        titles = row['titles'],
                        confidence = {'ACL': rec_valid['conf_ACL'], 'Loans': rec_valid['conf_Loans']}
                    )
                    df_Meta[col].loc[rec.cik] = rec

                acl = dfsACL[key]['ACL_num']
                df_ACL = df_ACL.join(acl, how='outer') 
                df_ACL.rename(columns={'ACL_num':'ACL'+'|'+key}, inplace=True)

                loans = dfsLoans[key]['Loans_num']
                df_Loans = df_Loans.join(loans, how='outer') 
                df_Loans.rename(columns={'Loans_num':'Loans'+'|'+key}, inplace=True)

                ratio = acl / loans
                ratio.name = 'Ratio'
                df_Ratio = df_Ratio.join(ratio, how='outer') 
                df_Ratio.rename(columns={'Ratio':'Ratio'+'|'+key}, inplace=True)

        df_Meta = df_Meta.iloc[::-1]

        # format output index labels
        df_ACL['cik'] = df_ACL.index
        def format_bank_name(val):
            for firm in LIST_ALL_FIRMS:
                if str(firm.get_info('cik')) == val:
                    name = firm.get_info('name').replace('\\','').replace('\/','')
                    if name.isupper():
                        name = name.title()
                    if firm.Scope == 'In':
                        return f"{name} - {firm.get_info('ticker')}"
                    else:
                        return f"(Out Of Scope) {name} - {firm.get_info('ticker')}"

        bank = df_ACL['cik'].apply(lambda x: format_bank_name(x))
        df_ACL.insert(0, "Bank", bank)
        df_ACL.drop(columns='cik', inplace=True)

        # apply minor corrections
        cols = df_ACL.columns.tolist()
        cols.remove('Bank')                                  
        for col in cols:
            df_ACL.loc[df_ACL[col] < 100, col] = np.nan

        list_of_df_tables = [df_Meta, df_ACL, df_Loans, df_Ratio]
        return list_of_df_tables



    def report_accounting_policy_completion(self, list_of_df_tables, output_path=False):
        """Complete report creation for accounting policy by outputing an excel file."""

        df_Meta, df_ACL, df_Loans, df_Ratio = list_of_df_tables
        qtrs = [col.split('meta|')[1] for col in df_Meta.columns]

        def create_xlsx_section(worksheet, row_start, col_start, df_col_offset, df, df_meta, acct_topic, qtrs, hdr_title):
            # config
            url = 'www.sec.gov/Archives/edgar/data/{cik}/{accn_wo_dash}/{accn}-index.htm'
            hdr_rows = 2
            time_periods = len(qtrs)
            col_end = col_start + time_periods
            row_end = row_start + df.shape[0]
            data_row_start = row_start + hdr_rows
            data_row_end = row_end + hdr_rows

            # header
            worksheet.set_column(col_start, col_end - 1, 11.5)          #header requires `-1` because it is inclusive
            worksheet.set_row(row_start, 20)
            worksheet.set_row(row_start + 1, 20)
            worksheet.merge_range(row_start, col_start, row_start, col_end-1, hdr_title, header_format)
            for idx, qtr in enumerate( qtrs ):
                col = col_start + idx
                worksheet.write_string(1, col, qtr, header_format)

            # data cells
            for idxC, col in enumerate( range(col_start, col_end)):
                for idxR, data_row in enumerate( range(data_row_start, data_row_end)):
                    # prepare
                    raw_value = df.iloc[idxR, idxC + df_col_offset]
                    rec = df_meta.iloc[idxR, idxC]
                    tmp_titles = ast.literal_eval(rec.titles) if (rec and rec.titles) else None
                    selection = [item for item in tmp_titles if item[0]==acct_topic] if tmp_titles else None
                    titles = selection[0] if selection else None
                    meta_value = {'cik':'None', 'accn':'None', 'form':'None', 'title':'None', 'xbrl':'None', 'confidence':'None'}
                    if rec:
                        meta_value['cik'] = rec.cik           #df_meta.index.tolist()[idxR]
                        meta_value['accn'] = rec.accn
                        meta_value['form'] = rec.form
                        meta_value['confidence'] = rec.confidence[acct_topic]
                        if titles:
                            meta_value['title'] = titles[1]
                            meta_value['xbrl'] = titles[2]
                    # formatting
                    if raw_value > 1:
                        value = raw_value / 1000000
                        data_format = workbook.add_format({'num_format': '#,##0.0', 'border':1})
                    else:
                        value = raw_value
                        data_format = workbook.add_format({'num_format': '0.000', 'border':1})
                    # write
                    if pd.isna(value): 
                        url_filled = url.format(cik=meta_value['cik'], accn_wo_dash=meta_value['accn'].replace('-',''), accn=meta_value['accn'])
                        comment = f'Form: {meta_value["form"]} \nTitle: {meta_value["title"]} \nXBRL: {meta_value["xbrl"]} \nValidated: {meta_value["confidence"]} \nDoc url: {url_filled}'
                        worksheet.write_string(data_row, col, '-', missing_format)
                        worksheet.write_comment(data_row, col, comment, comment_format)
                    else:
                        url_filled = url.format(cik=meta_value['cik'], accn_wo_dash=meta_value['accn'].replace('-',''), accn=meta_value['accn'])
                        comment = f'Form: {meta_value["form"]} \nTitle: {meta_value["title"]} \nXBRL: {meta_value["xbrl"]} \nValidated: {meta_value["confidence"]} \nDoc url: {url_filled}'
                        worksheet.write_number(data_row, col, value, data_format)
                        worksheet.write_comment(data_row, col, comment, comment_format)


        # xlsx report output
        file_path = self.output_path / 'report_acl_acct.xlsx'
        #file_path = './archive/report/report_acl_acct.xlsx'
        workbook = xlsxwriter.Workbook(file_path)
        worksheet = workbook.add_worksheet('Large Banks')
        docsheet = workbook.add_worksheet('Documentation')

        # worksheet
        banks = df_ACL.Bank.tolist()
        col_start = 2
        row_start = 0
        df_col_offset = 1                               #shift one column for banks
        section2_col_start = col_start + 1 * len(qtrs) + 1 * 1
        section3_col_start = col_start + 2 * len(qtrs) + 2 * 1

        header_format = workbook.add_format({
            'bold': 1,
            'border': 1,
            'bg_color': '#060a7d',
            'font_color': 'white',
            'font_size': 14,
            'align': 'center',
            'valign': 'vcenter'})
        index_format = workbook.add_format({
            'text_wrap': True,
            'bold': 1,
            'border': 1,
            'bg_color': '#060a7d',
            'font_color': 'white',
            'align': 'right'
        })
        missing_format = workbook.add_format({
            'border': 1,
            'align': 'center',
            'font_color': 'gray'
            })
        comment_format = {
            'visible': False,
            'width': 200,
            'height': 100,
            'color': '#f7f7f5'
            }

        # index rows with bank names
        worksheet.set_column(1, 1, 35)
        for idx, name in enumerate(banks):
            row = idx + 2
            worksheet.write_string(row, 1, name,  index_format)

        # create sections
        create_xlsx_section(worksheet, row_start, col_start, df_col_offset, df_ACL, df_Meta, acct_topic='ACL', qtrs=qtrs, hdr_title='Allowance for Credit Losses')
        df_col_offset = 0
        create_xlsx_section(worksheet, row_start, section2_col_start, df_col_offset, df_Loans, df_Meta, acct_topic='Loans', qtrs=qtrs, hdr_title='Loans')
        create_xlsx_section(worksheet, row_start, section3_col_start, df_col_offset, df_Ratio, df_Meta, acct_topic='ACL', qtrs=qtrs, hdr_title='Coverage')
        

        # docsheet
        for idx, doc in enumerate(doc_records):
            for col, val in enumerate(doc.items()):
                if val[1]:
                    docsheet.write_string(idx+1, col, val[1])
        for col, val in enumerate(doc.items()):
            if val[0]:
                docsheet.write_string(0, col, val[0])
        

        # return
        workbook.close()
        return df_ACL



    def trend(self, df_ACL, output_path):
        '''Create trend lines
        TODO: this seems to re-create df_long?  maybe just use that?
        '''
        cols = df_ACL.columns.tolist()
        cols.pop(0)   #remove 'Bank'
        df = pd.melt(df_ACL, id_vars='Bank', value_vars=cols)
        df.rename(columns={'value':'ACL'}, inplace=True)
        df['yr_qtr'] = df['variable'].str.split('|').str[1]
        df['dt_filed'] = pd.PeriodIndex(df['yr_qtr'], freq='Q').to_timestamp()
        df['ACL'] = df.ACL.astype(float)

        years = 2
        start = date.today() + timedelta(days=30)
        end = date.today() - timedelta(days=365*years)
        plt = (ggplot(aes(x='dt_filed', y='ACL'), df) 
                + geom_point(aes(color='Bank'), alpha=0.7)
                + geom_line(aes(color='Bank')) 
                + scale_x_datetime(labels=date_format('%Y-%m'), limits=[end, start]) 
                #+ scale_y_log10()
                + scale_y_continuous(limits=[0, 20000])
                + labs(y='Allowance for credit losses', 
                        x='Date', 
                        title="Firms' Allowance for Credit Losses over Time")
                + theme(figure_size=(12, 6))
                )
        plt_file_path = output_path / 'trend.jpg'
        plt.save(filename = plt_file_path, height=3, width=9, units = 'in', dpi=250)
        return True



    def template(self, df_long, dir_path=False):
        start = date.today() + timedelta(days=30)
        end = date.today() - timedelta(days=365*3)
        plt = (ggplot(aes(x='dt_filed', y='ACL'), df_long) 
                + geom_point(aes(color='cik'), alpha=0.7)
                + geom_line(aes(color='cik')) 
                + scale_x_datetime(labels=date_format('%Y-%m'), limits=[end, start]) 
                + scale_y_log10()
                + labs(y='log Allowance for credit losses', 
                        x='Date', 
                        title="Firms' Allowance for Credit Losses over Time")
                + theme(figure_size=(12, 6))
                )
        plt_file_path = dir_path / 'trend.jpg'
        plt.save(filename = plt_file_path, height=3, width=9, units = 'in', dpi=250)
        with open(plt_file_path, 'rb') as f:
            b64 = base64.b64encode( f.read() )
        b64_string = b64.decode('ascii')
        image_src = '"data:image/jpg;base64, ' + b64_string + '"'
        page_title_text = 'ACL Trend Report'
        title_text = '<placeholder>'
        text = '<placeholder>'
        html = f'''
            <html>
                <head>
                    <title>{page_title_text}</title>
                </head>
                <body>
                    <h1>{title_text}</h1>
                    <p>{text}</p>
                    <img src={image_src} width="1200">
                    <p>{text}</p>
                    {df_long.to_html()}
                </body>
            </html>
            '''
        file_path = dir_path / 'report_trend.html'
        with open(file_path, 'w') as f:
            f.write(html)
        options = {
            "enable-local-file-access": True
            }
        try:
            pdfkit.from_file(str(file_path), './archive/report/report_trend.pdf')
        except:
            pass
        return True


    def validate(self, df_long, dir_path=False):
        """Validate 8-K earnings against associated 10-K/-Q quarterly statements.
        If not exact, then these numbers should be close.
        """
        def get_confidence(df, acct):
            old_col = 'diff_'+acct
            new_col = old_col.replace('diff_','conf_')
            result = df[['cik','yr_qtr_10q']]
            result[new_col] = 0
            for cik in df.value_counts('cik').index:
                cols = ['cik', 'yr_qtr_10q', old_col]                          #[['cik', 'yr_qtr_10q', 'ACL_10q', 'ACL_8k', acct]]
                tmp = df[df['cik']==cik][cols].copy('deep')
                tmp[new_col] = 1 - (1 / (1 + np.exp( -abs(tmp[old_col]) )))    # 1-sigmoid with range: 0 - inf
                tmp[new_col][tmp[old_col] == 0] = 1
                tmp[new_col][tmp[old_col].isna() == True] = 0
                #tmp[new_col][tmp[new_col] != 0] = tmp[acct].sum() / tmp[acct].__len__()
                result.loc[tmp.index, new_col] = tmp[new_col]
            conf = result[new_col]
            return conf

        #prepare
        suffixes = ('_10q', '_8k')
        df_qtrly = df_long[df_long['form'].isin(['10-K','10-Q'])]
        df_qtrly['fp'].replace({'FY':'Q4'}, inplace=True)
        df_8k = df_long[~df_long['form'].isin(['10-K','10-Q'])]
        df_tmp = pd.merge(df_qtrly, df_8k, on=['fy','fp','cik'], how='left', suffixes=suffixes)

        #process for each account
        for acct in LIST_ALL_ACCOUNTS:
            left = acct+suffixes[0]
            right = acct+suffixes[1]
            df_tmp[left] = df_tmp[left].replace(r'^([A-Za-z]|[0-9]|_)+$', np.NaN, regex=True).abs()
            df_tmp[right] = df_tmp[right].replace(r'^([A-Za-z]|[0-9]|_)+$', np.NaN, regex=True).abs()
            col_diff = {'_spacer'+acct: ' - ',
                        '_'+left: df_tmp[left],
                        '_'+right: df_tmp[right],
                        'diff_'+acct: df_tmp[left] - df_tmp[right],
                        'conf_'+acct: 0
                        }
            df_tmp = df_tmp.assign(**col_diff)
            df_tmp['conf_'+acct] = get_confidence(df = df_tmp, acct = acct)
            #df_tmp.drop(columns=[left, right], inplace=True)

        #output
        cols_merged = ['cik', 'yr_qtr_10q', 'accn_10q','accn_8k', 'ACL_10q','ACL_8k', 'conf_ACL', 'Loans_10q', 'Loans_8k', 'conf_Loans']
        cols_rename = {'yr_qtr_10q':'yr_qtr'}
        df_valid = df_tmp[cols_merged].rename(columns = cols_rename)

        file_path = Path(dir_path) / 'report_validation.csv'
        df_valid.to_csv(file_path, index=False)
        return df_valid