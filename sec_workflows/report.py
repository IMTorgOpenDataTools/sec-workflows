#!/usr/bin/env python3
"""
Module Docstring
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"




#third-party
import xlsxwriter
import pdfkit

import pandas as pd
import sqlalchemy as sql                     #create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData
import sqlalchemy_utils as sql_util         #database_exists, create_database

from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

#built-in
from asyncio.log import logger
from pathlib import Path
import logging
import time
from datetime import datetime, date, timedelta
import requests
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
    accts
)




class Report:
    """Report class to output various types from database."""

    def __init__(self, db, output_path):
        self.db = db
        self.output_path = output_path


    def create_report(self, type):
        """Primary entry to create new report"""
        dir_path = Path(self.output_path).parent
        match type:
            case 'long': 
                result = self.report_long(self.output_path)
            case 'accounting_policy':
                df_long = self.report_long() 
                result = self.report_accounting(df_long, dir_path)
            case 'trend':
                df_long = self.report_long()
                #result = template(df_long, dir_path)
                df_ACL = self.report_accounting(df_long, dir_path)
                result = self.trend(df_ACL, dir_path)
            case 'validate':
                df_long = self.report_long()
                result = self.validate(df_long, dir_path)

        return result



    def report_long(self, output_path):
        """TODO: move to db query.  It is necessary for everything, so just make it automatic."""
        df = self.db.query_database(table_name='filings')
        df['dt_filed'] = pd.to_datetime(df.filed)
        df['form'] = pd.Categorical(df['form'], 
                                    categories=['8-K/EX-99.1','10-Q','10-K'], 
                                    ordered=True)
        df.sort_values(by=['cik','yr_qtr','form'], inplace=True, ascending=False)
        if df.shape[0] > 0 and output_path:
            df.to_csv(output_path, index=False)
            logger.info(f'Report saved to path: {output_path}')
        else:
            logger.info(f'No data available to report')
        return df


    def report_accounting(df_long, output_path=False):
        """Report quarterly accounts only using 8-K extractions."""

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

        # get df for each qtr
        df_result = pd.DataFrame()
        df_result['cik'] = df_long['cik'].unique()
        df_result.set_index('cik', inplace=True)
        dfs = {}
        for idx,qtr in enumerate(qtrs):                 #TODO: create columns, individually, to get the best score for ACL, then Loans
            df_tmp1 = df_tmp[df_tmp['yr_qtr'] == qtr]
            df_tmp2 = df_tmp1.sort_values(by=['cik','form'], ascending=False).dropna(subset=['ACL']).drop_duplicates(subset='cik')
            df_tmp3 = df_tmp2.groupby('cik').head(1)
            df_tmp3.set_index('cik', inplace=True)
            if df_tmp3.shape[0] > 0:
                dfs[qtr] = df_tmp3

        # create df by adding columns for each qtr
        meta = namedtuple('meta_record', ['accn', 'form', 'titles'])
        df_Meta = df_result.copy(deep=True)
        df_ACL = df_result.copy(deep=True)
        df_Loans = df_result.copy(deep=True)
        df_Ratio = df_result.copy(deep=True)
        for key in dfs.keys():
            if dfs[key].shape[0] > 0:

                col = 'accn'+'|'+key
                df_Meta[col] = None
                for idx, row in enumerate(dfs[key].to_dict('records')):
                    rec = meta(
                        accn = row['accn'],
                        form = row['form'],
                        titles= row['titles']
                    )
                    df_Meta[col].iloc[idx] = rec

                acl = dfs[key]['ACL_num']
                df_ACL = df_ACL.join(acl, how='outer') 
                df_ACL.rename(columns={'ACL_num':'ACL'+'|'+key}, inplace=True)

                loans = dfs[key]['Loans_num']
                df_Loans = df_Loans.join(loans, how='outer') 
                df_Loans.rename(columns={'Loans_num':'Loans'+'|'+key}, inplace=True)

                ratio = acl / loans
                ratio.name = 'Ratio'
                df_Ratio = df_Ratio.join(ratio, how='outer') 
                df_Ratio.rename(columns={'Ratio':'Ratio'+'|'+key}, inplace=True)

        # format output
        df_ACL['cik'] = df_ACL.index
        def format_bank_name(val):
            for firm in firms:
                if str(firm.get_info('cik')) == val:
                    name = firm.get_info('name')
                    if name.isupper():
                        name = name.title()
                    if firm.Scope == 'In':
                        return f"{name} - {firm.get_info('ticker')}"
                    else:
                        return f"(Out Of Scope) {name} - {firm.get_info('ticker')}"

        bank = df_ACL['cik'].apply(lambda x: format_bank_name(x))
        df_ACL.insert(0, "Bank", bank)
        df_ACL.drop(columns='cik', inplace=True)

        cols = df_ACL.columns.tolist()
        cols.pop(0)   #remove 'Bank'
        # if incorrect, then just make NaN
        for col in cols:
            df_ACL.loc[df_ACL[col] < 100, col] = np.nan
        #df_ACL = df_ACL[df_ACL['Bank']!='BKU']              #TODO:change once updated, but check first

        df_list = [df_ACL, df_Ratio]
        df_result = df_result.join(df_list[0])
        df_result = df_result.join(df_list[1])

        #df_long[df_long.cik == '91576']
        file_path = output_path / 'report_acl_acct.csv'
        df_result.to_csv(file_path)


        def create_xlsx_section(worksheet, row_start, col_start, df_col_offset, df, df_meta, qtrs, hdr_title):
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
                    titles = ast.literal_eval(df_meta.iloc[idxR, idxC].titles) if (rec and rec.titles) else None
                    meta_value = {'cik':'None', 'accn':'None', 'form':'None', 'title':'None', 'xbrl':'None'}
                    if rec:
                        meta_value['cik'] = df_meta.index.tolist()[idxR]
                        meta_value['accn'] = rec.accn
                        meta_value['form'] = rec.form
                        if titles:
                            meta_value['title'] = ast.literal_eval(rec.titles)[0][1]
                            meta_value['xbrl'] = ast.literal_eval(rec.titles)[0][2]
                    if raw_value > 1:
                        value = raw_value / 1000000
                        data_format = workbook.add_format({'num_format': '#,##0.0', 'border':1})
                    else:
                        value = raw_value
                        data_format = workbook.add_format({'num_format': '0.000', 'border':1})
                    # write
                    if pd.isna(value): 
                        url_filled = url.format(cik=meta_value['cik'], accn_wo_dash=meta_value['accn'].replace('-',''), accn=meta_value['accn'])
                        comment = f'Form: {meta_value["form"]} \nTitle: {meta_value["title"]} \nXBRL: {meta_value["xbrl"]} \nconfidence: 0 \ndoc url: {url_filled}'
                        worksheet.write_string(data_row, col, '-', missing_format)
                        worksheet.write_comment(data_row, col, comment, comment_format)
                    else:
                        url_filled = url.format(cik=meta_value['cik'], accn_wo_dash=meta_value['accn'].replace('-',''), accn=meta_value['accn'])
                        comment = f'Form: {meta_value["form"]} \nTitle: {meta_value["title"]} \nXBRL: {meta_value["xbrl"]} \nconfidence: 1 \ndoc url: {url_filled}'
                        worksheet.write_number(data_row, col, value, data_format)
                        worksheet.write_comment(data_row, col, comment, comment_format)


        # xlsx report
        file_path = output_path / 'report_acl_acct.xlsx'
        workbook = xlsxwriter.Workbook(file_path)
        worksheet = workbook.add_worksheet('Large Banks')

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

        # sections
        create_xlsx_section(worksheet, row_start, col_start, df_col_offset, df_ACL, df_Meta, qtrs, hdr_title='Allowance for Credit Losses')
        df_col_offset = 0
        create_xlsx_section(worksheet, row_start, section2_col_start, df_col_offset, df_Loans, df_Meta, qtrs, hdr_title='Loans')
        create_xlsx_section(worksheet, row_start, section3_col_start, df_col_offset, df_Ratio, df_Meta, qtrs, hdr_title='Coverage')
        workbook.close()


        # return
        return df_ACL


    def trend(df_ACL, output_path):
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



    def report_accounting_certified(df_long, output_path=False):
        """FUTURE: Report of quarterly accounts using strict stability requirements.

        Comparability requirements (currently deficient):
        * 8-K data should only be used when certified 10-K/-Q data is not available (quarterly is incorrect because of inconsistency in reporting).  ​
        * Certified 10-K/-Q data should be both stable and most-current (this may not be true for some accounts). ​
        * Accounts between firms should be 'reasonably' comparable with descriptions documented (references needed).​

        """
        ciks = df_long.cik.unique().tolist()
        qtrly = ['10-K','10-Q']
        days_from_last_qtr = [0, 90, 180, 270, 360]
        df_long['yr_qtr'] = df_long.apply(create_qtr, axis=1)

        df_result = pd.DataFrame()
        df_result['cik'] = df_long['cik'].unique()
        df_result.set_index('cik', inplace=True)
        dfs = {}
        for days in days_from_last_qtr:
            prev = pd.Timestamp(datetime.now())  -  pd.to_timedelta( days, unit='d')
            prev_qtr = f'{prev.year}-{prev.quarter}'
            # try 10-K/-Q first
            # mask = ~df_long['form'].isin( qtrly ) if days == 90 else df_long['form'].isin( qtrly )
            # df_long[ (df_long['yr_qtr'] == '2021-4') & (~df_long['form'].isin( qtrly ))]
            mask = df_long['form'].isin( qtrly )
            df_10k_qtr = df_long[ (df_long['yr_qtr'] == prev_qtr) & (pd.isna(df_long['ACL'])==False) & (mask)]
            ciks_10k = df_10k_qtr.cik.unique()
            if len(ciks_10k) < len(ciks):
                mask = ~df_long['form'].isin( qtrly )
                df_8k_qtr = df_long[ (df_long['yr_qtr'] == prev_qtr) & (pd.isna(df_long['ACL'])==False)  & (mask)]
                if len(ciks_10k) > 0:
                    s8k = set(df_8k_qtr.cik)
                    s10k = set(df_10k_qtr.cik)
                    diff = list( s8k.difference(s10k) )
                    df_10k_tmp = df_10k_qtr.drop_duplicates(subset=['cik'], inplace=False)
                    df_8k_tmp = df_8k_qtr[df_8k_qtr.cik.isin(diff)].drop_duplicates(subset=['cik'], inplace=False)
                    df_qtr = df_10k_tmp.append(df_8k_tmp)    #, ignore_index=True
                else:
                    df_qtr = df_8k_qtr
            else:
                df_qtr = df_10k_qtr

            df_qtr.drop_duplicates(subset=['cik'], inplace=True)
            df_qtr.set_index('cik', inplace=True)
            dfs[prev_qtr] = df_qtr

        df_ACL = df_result.copy(deep=True)
        df_Loans = df_result.copy(deep=True)
        df_Ratio = df_result.copy(deep=True)
        for key in dfs.keys():
            if dfs[key].shape[0] > 0:
                acl = np.abs( dfs[key]['ACL'] )
                df_ACL = df_ACL.join(acl, how='outer') 
                df_ACL.rename(columns={'ACL':'ACL'+'|'+key}, inplace=True)

                loans = dfs[key]['Loans']
                df_Loans = df_Loans.join(loans, how='outer') 
                df_Loans.rename(columns={'Loans':'Loans'+'|'+key}, inplace=True)

                ratio = acl / loans
                ratio.name = 'Ratio'
                df_Ratio = df_Ratio.join(ratio, how='outer') 
                df_Ratio.rename(columns={'Ratio':'Ratio'+'|'+key}, inplace=True)

        df_list = [df_ACL, df_Ratio]
        df_result = df_result.join(df_list[0])
        df_result = df_result.join(df_list[1])
        #df_long[df_long.cik == '91576']
        file_path = './archive/report/report_acl_acct.csv'
        df_result.to_csv(file_path, index=False)
        return df_ACL


    def template(df_long, dir_path=False):
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


    def validate(df_long, dir_path=False):
        """Validate 8-K earnings against associated 10-K/-Q quarterly statements.
        If not exact, then these numbers should be close.
        """
        df_qtrly = df_long[df_long['form'].isin(['10-K','10-Q'])]
        df_8k = df_long[~df_long['form'].isin(['10-K','10-Q'])]
        df_tmp = pd.merge(df_qtrly, df_8k, on=['fy','fp','cik'], how='left')
        for acct in accts:
            left = acct+'_x'
            right = acct+'_y'
            df_tmp[left] = df_tmp[left].replace(r'^([A-Za-z]|[0-9]|_)+$', np.NaN, regex=True)
            df_tmp[right] = df_tmp[right].replace(r'^([A-Za-z]|[0-9]|_)+$', np.NaN, regex=True)
            df_tmp[acct+'_diff'] = df_tmp[left] - df_tmp[right]
            #df_tmp.drop(columns=[left, right], inplace=True)
        df_valid = df_tmp
        file_path = dir_path / 'report_validation.csv'
        df_valid.to_csv(file_path, index=False)
        return True