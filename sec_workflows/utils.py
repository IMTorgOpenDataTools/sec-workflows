#!/usr/bin/env python3
"""
Module Docstring
"""
from asyncio.log import logger
from collections import namedtuple
import ast
from enum import unique
import pandas as pd
import numpy as np
import sqlalchemy as sql
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

from pathlib import Path
import time
from datetime import datetime, date, timedelta
import re

import xlsxwriter
import pdfkit
import base64

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http

from sec_edgar_downloader import Downloader
from sec_edgar_downloader import UrlComponent as uc
from sec_edgar_extractor.extract import Extractor
from sec_edgar_extractor.instance_doc import Instance_Doc

import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    log_file,
    sec_edgar_downloads_path,
    db_file,
    table_name,
    MAX_RETRIES,
    SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,
    DEFAULT_TIMEOUT,
    MINUTES_BETWEEN_CHECKS,
    QUARTERS_IN_TABLE,
    accts,
    #config,
    meta,
    filings,
    RecordMetadata,
    FilingMetadata
)

url_firm_details = "https://data.sec.gov/submissions/CIK{}.json"
#url_api_account = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'     #long_cik, acct

# Specify max number of request retries
# https://stackoverflow.com/a/35504626/3820660
retries = Retry(
    total=MAX_RETRIES,
    #backoff_factor=SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,        # sleep between requests with exponentially increasing interval: {backoff factor} * (2 ** ({number of total retries} - 1))
    status_forcelist=[403, 500, 502, 503, 504],
)

#http.client.HTTPConnection.debuglevel = 1                      # display more information 

class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)







def load_firms(file_path):
    """Load file containing firms."""
    df = pd.read_csv(file_path)
    firm_recs = df.to_dict('records')
    firms = []
    for firm in firm_recs:
        if firm:
            item = uc.Firm(ticker = firm['Ticker'] )
            firms.append( item )
    return firms


 
def api_request(session, type, cik, acct):
    """Make a request with a given session.
    TODO: replace with Downloader._utils.request_standard_url(payload, headers)
    """
    #prepare request
    headers = {
        'User-Agent': 'YoMmama',
        'From': 'joe.blo@gmail.com',
        'Accept-Encoding': 'gzip, deflate'
        }
    url_firm_details = "https://data.sec.gov/submissions/CIK{}.json"
    url_company_concept = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'     #long_cik, acct

    long_cik = make_long_cik(cik)
    match type:
        case 'firm_details': 
            filled_url = url_firm_details.format(long_cik)
            keys = ('filings','recent')
        case 'concept': 
            filled_url = url_company_concept.format(long_cik, acct)
            keys = ('units','USD')
        case _:
            logger.warning(f'`api_request` type={type} is not an option (`firm_detail`, `concept`)')
            exit()

    #make request
    edgar_resp = session.get(filled_url, headers=headers)
    if edgar_resp.status_code == 200:
        if keys[0] in edgar_resp.json().keys():
            if keys[1] in edgar_resp.json()[keys[0]].keys():
                items = edgar_resp.json()[keys[0]][keys[1]]
                return items
    else:
        logger.warning(f'edgar response returned status code {edgar_resp.status_code}')
        return None



def remove_list_dups(lst, key):
    """Remove objects in a list if their key is already present (ordered)."""
    seen = set()
    new_lst = []
    for d in lst:
        t = tuple(d[key])
        if t not in seen:
            seen.add(t)
            new_lst.append(d)
    return new_lst




#for each cik:
# get the most-recent 8-K/99.* filing
# extract accts 
# load into sqlite
#add 8-K acct that is newer than 10-k
def get_recent_financial_release(tickers, ciks):
    """Search and download the most-recent 8-K/EX-99.* documents."""
    def check(row): 
        return 'EX-99.1' in row.Type if type(row.Type) == str else False   #TODO:add 99.2+

    start = date.today() - timedelta(days=90)
    banks = [uc.Firm(ticker=bank[1]) for bank in tickers]
    dl = Downloader(sec_edgar_downloads_path)
    for bank in banks:
        ticker = bank.get_info()['ticker']
        urls = dl.get_urls("8-K",
                            ticker,
                            after = start.strftime("%Y-%m-%d")
                            )
    df = dl.filing_storage.get_dataframe(mode='document')
    sel1 = df[(df['short_cik'].isin(ciks)) & (df['file_type'] == '8-K') & (df['FS_Location'] == '')]
    mask = sel1.apply(check, axis=1)
    sel2 = sel1[mask] 
    lst_of_idx = sel2.index.tolist()
    staged = dl.filing_storage.get_document_in_record( lst_of_idx )
    downloaded_docs = dl.get_documents_from_url_list(staged)
    return downloaded_docs


def extract_accounts_from_documents(docs):
    for doc in docs:
        doc.FS_Location
        #TODO:extraction module
    pass


def make_long_cik(cik):
    short_cik = str(cik)
    if len(short_cik) < 10:
        zeros_to_add = 10 - len(short_cik)
        long_cik = str(0) * zeros_to_add + str(short_cik)
    else:
        long_cik = short_cik
    return long_cik


def scale_value(val, scale):
    match scale:
        case 'thousands':
            result_val = val * 1000
        case 'millions':
            result_val = val * 1000000
        case 'billions':
            result_val = val * 1000000000
        case _:
            result_val = val
    return result_val



def initialize_db(db, firms):
    """Initialize the database with certified quarterly earnings (10-K/-Q).
    """
    # configure
    FILE_TYPES = ['10-K', '10-Q']
    AFTER = "2021-01-01"

    path_download = "./archive/downloads"
    dl = Downloader(path_download)
    ex = Extractor(save_intermediate_files=True)
    idoc = Instance_Doc()

    # download
    ciks = [str(firm._cik) for firm in firms]
    tickers = [(str(firm._cik), str(firm._ticker)) for firm in firms]
    for firm in firms:
        for filing in FILE_TYPES:
            TICKER = firm.get_info()['ticker']
            urls = dl.get_urls(filing,
                                TICKER, 
                                after=AFTER
                                )   
    df_fs = dl.filing_storage.get_dataframe(mode='document')
    sel1 = df_fs[(df_fs['short_cik'].isin(ciks)) 
            & (df_fs['file_type'].isin(FILE_TYPES))
            & (df_fs['Type'] == 'XML')
            & (df_fs['Document'].str.contains('_htm.xml', case=True))
            ]
    lst_of_idx = sel1.index.tolist()
    staged = dl.filing_storage.get_document_in_record( lst_of_idx )
    updated_docs = dl.get_documents_from_url_list(staged)

    # prepare and load data items
    selected_docs = list(updated_docs['new']) + list(updated_docs['previous'])
    recs = []
    for key, doc in selected_docs:
        if not doc.FS_Location:
            continue
        cik = doc.FS_Location.parent.parent.name
        ticker = [item[1] for item in tickers if item[0] == cik][0]
        accn = doc.FS_Location.parent.name
        key = f'{cik}|{accn}'
        filing = dl.filing_storage.get_record(key)
        accts = ex.config[ ticker ].accounts.items()

        start = pd.to_datetime( filing.file_date )
        with open(doc.FS_Location, 'r') as f:
            file_htm_xml = f.read()
        df_doc, df = idoc.create_xbrl_dataframe( file_htm_xml )
        fy = df_doc.value[df_doc['name']=='DocumentFiscalYearFocus'].values[0]
        fp = df_doc.value[df_doc['name']=='DocumentFiscalPeriodFocus'].values[0]
        end = df_doc.value[df_doc['name']=='DocumentPeriodEndDate'].values[0]
        for acct_key, acct_rec in accts:
            try:
                xbrl_tag = acct_rec.xbrl                        #<<< try a list of similar xbrls until one hits
                item = df[(df['concept'] == xbrl_tag )               
                            & (df['dimension'] == '' ) 
                            & (df['value_context']== '' )
                            & (df['start'] < start )
                            & (pd.isna(df['end']) == True )
                            ].sort_values(by='start', ascending=False).to_dict('records')[0]
            except:
                continue
            rec = RecordMetadata(
                    cik = cik,
                    accn = accn,
                    form = filing.file_type,
                    account = acct_key,
                    value = item['value_concept'],                  #not necessary: scale_value(item["val"], acct_val.scale),
                    account_title = acct_rec.table_account,
                    xbrl_tag = xbrl_tag,
                    fy = fy,
                    fp = fp,
                    end = end,
                    filed = filing.file_date
            )
            recs.append( rec )


    # prepare records and insert into db
    if len(recs) > 0:
        df = pd.DataFrame(recs, columns=RecordMetadata._fields)
        df_columns = df.drop(labels=["account","value","account_title","xbrl_tag"], axis=1).drop_duplicates(subset=["cik","accn"])
        df_wide = df.pivot(index=["cik","accn"], columns="account", values="value").reset_index()

        tmp_key = pd.DataFrame(  df.groupby(['cik','accn'])['account'].apply(list) )
        tmp_key['account_title'] = pd.DataFrame(  df.groupby(['cik','accn'])['account_title'].apply(list) )['account_title']
        tmp_key['xbrl_tag'] = pd.DataFrame(  df.groupby(['cik','accn'])['xbrl_tag'].apply(list) )['xbrl_tag']
        tmp_key['titles'] = tmp_key.apply(lambda row: str(list( zip(row['account'], row['account_title'], row['xbrl_tag']))), axis=1 )
        tmp_key['index'] = tmp_key.index
        tmp_key.reset_index(inplace=True)
        df_wide['titles'] = tmp_key['titles']

        df_wide_total = pd.merge(df_wide, df_columns, on=["cik","accn"], how="left")
        df_wide_total["filed"] = pd.to_datetime(df_wide_total["filed"])
        df_wide_total.sort_values(by="filed", ascending=False, inplace=True)
        #TODO: apply FilingMetadata before importing
        try:
            df_wide_total.to_sql(db.table_name, 
                    con=db.engine,
                    if_exists="append", 
                    index=False
                    )
        except sql.exc.IntegrityError as e:
            logger.warning("Unique key violation on insert")
        else:
            logger.info(f"Inserted {df_wide_total.shape[0]} records to table {db.table_name}")
        return True
    else:
        return False








'''
def initialize_db(db, firms):
    """Initialize the database.

    TODO: scale values by config units (usually millions)
    """
    #request and populate records
    extractor = Extractor(config=config,
                            save_intermediate_files=True
                            )
    with requests.Session() as client:
        client.mount("http://", TimeoutHTTPAdapter(max_retries=retries))
        client.mount("https://", TimeoutHTTPAdapter(max_retries=retries))
        client.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0"    # mimick the browser
            })

        recs = []
        for firm in firms:
            logger.info(f"get accounts for firm: {firm._name}")
            if not firm._ticker in extractor.config.keys():
                continue
            accts = extractor.config[ firm._ticker ].accounts.items()
            for acct_key, acct_val in accts:
                logger.info(f"api request for account: {acct_key}")
                items = api_request(session=client, type="concept", cik=firm._cik, acct=acct_val.xbrl)      # if custom xbrl acct fails, then try default
                if not items:
                    continue
                items.reverse()
                items_dedup = remove_list_dups(items, "accn")
                items_dedup.reverse()
                end = len(items_dedup)
                idx_start = end - QUARTERS_IN_TABLE if QUARTERS_IN_TABLE < end  else end 
                tgt_items = items_dedup[idx_start:]
                for item in tgt_items:
                    rec = FilingMetadata(
                            cik = firm._cik,
                            accn = item["accn"],
                            form = item["form"],
                            acct = acct_key,
                            val = item["val"],                  #not necessary: scale_value(item["val"], acct_val.scale),
                            fy = item["fy"],
                            fp = item["fp"],
                            end = item["end"],
                            filed = item["filed"]
                    )
                    recs.append( rec )
                time.sleep(SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL)

    #prepare records and insert into db
    if len(recs) > 0:
        df = pd.DataFrame(recs, columns=FilingMetadata._fields)
        df_columns = df.drop(labels=["acct","val"], axis=1).drop_duplicates(subset=["cik","accn"])
        df_wide = df.pivot(index=["cik","accn"], columns="acct", values="val").reset_index()
        df_wide_total = pd.merge(df_wide, df_columns, on=["cik","accn"], how="left")
        df_wide_total["filed"] = pd.to_datetime(df_wide_total["filed"])
        df_wide_total.sort_values(by="filed", ascending=False, inplace=True)
        try:
            df_wide_total.to_sql(db.table_name, 
                    con=db.engine,
                    if_exists="append", 
                    index=False
                    )
        except sql.exc.IntegrityError as e:
            logger.warning("Unique key violation on insert")
            return False
        else:
            logger.info(f"Inserted {df_wide_total.shape[0]} records to table {db.table_name}")
            return True
    else:
        return False
'''




def get_press_releases(db, firms):
    """Get press releases (8-K/EX-99.*) and extract account values.
    """

    # supporting functions
    def get_8k_qtr(df8k, df10q):
        """Get the 8-K quarter by referencing the correspoding 10-Qs."""

        periods = ['Q1','Q2','Q3','FY','Q1']            #index() will only take the first occurrence
        quarters = {'Q1':1,'Q2':2,'Q3':3,'FY':4}

        #prepare
        df8k['dt_file_date'] = pd.to_datetime(df8k['filed'])
        df10q['dt_filed'] = pd.to_datetime(df10q['filed'])
        df8k.sort_values(by=['cik','dt_file_date'], inplace=True)
        df10q.sort_values(by=['cik','dt_filed'], inplace=True)
        df10q = df10q[df10q['form']!='8-K']
        ciks10q = df10q['cik'].tolist()

        #check each filing
        results = []
        for doc in df8k.to_dict('records'):
            cik = str(doc['cik']) 
            fd = doc['dt_file_date']
            if cik in ciks10q:
                dftmp = df10q[(df10q['cik'] == cik) & (df10q['dt_filed'] >= fd)]
                if dftmp.shape[0] > 0:
                    item = dftmp.iloc[0]['yr-qtr']
                    results.append( item )
                else:
                    dftmp = df10q[(df10q['cik'] == cik) & (df10q['dt_filed'] < fd)].sort_values(by='dt_filed', ascending=False)
                    last_record = dftmp.iloc[0]
                    period = last_record['fp']
                    idx = periods.index(period)
                    new_period = periods[idx + 1]
                    new_qtr = quarters[new_period]
                    new_year = last_record['fy'] if period in periods[:2] else int(last_record['fy']) + 1
                    item = f"{new_year}-{new_qtr}"
                    results.append( item )
            else: results.append( None )
        return results


    def create_qtr(row):
        """Create the column formatted `yr-qtr`."""
        if row['fp']=='FY': 
            fp = 4
        else:
            fp = str(row['fp']).replace('Q','')
        return str(row['fy']) + '-' + str(fp)


    def check(row):
        """Mask for selecting press release data."""
        if type(row.Type) == str and ('EX-99.1' in row.Type or 'EX-99.2' in row.Type or 'EX-99.3' in row.Type):
            return True
        else:
            return False


    def scale_value(val, scale):
        match scale:
            case 'thousands':
                result_val = val * 1000
            case 'millions':
                result_val = val * 1000000
            case 'billions':
                result_val = val * 1000000000
            case _:
                result_val = val
        return result_val


    path_download = "./archive/downloads"
    intermediate_step = Path('./archive/intermediate_dataframe.csv')            # TODO: this should be a db table
    if intermediate_step.is_file():
        df = pd.read_csv(intermediate_step)
    else:
        # download
        ciks = [str(firm._cik) for firm in firms]
        dl = Downloader(path_download)
        for firm in firms:
            TICKER = firm.get_info()['ticker']
            urls = dl.get_urls("8-K",
                                TICKER, 
                                after="2021-01-01")   
        df = dl.filing_storage.get_dataframe(mode='document')
        sel1 = df[(df['short_cik'].isin(ciks)) 
                    & (df['file_type'] == '8-K') 
                    ]
        mask = sel1.apply(check, axis=1)
        sel2 = sel1[mask]
        lst_of_idx = sel2.index.tolist()
        staged = dl.filing_storage.get_document_in_record( lst_of_idx )
        updated_docs = dl.get_documents_from_url_list(staged)
        selected_docs = list(updated_docs['new']) + list(updated_docs['previous'])


        # extract
        extractor = Extractor(save_intermediate_files=True)
        recs = []
        for key, doc in selected_docs:
            if not doc.FS_Location:
                continue
            cik = doc.FS_Location.parent.parent.name             #TODO:<<<fix this craziness by creating a class and providing attributes: cik, accn
            target_firm = [firm for firm in firms if firm.get_info()['cik'].__str__()==cik][0]
            ticker = target_firm.get_info()['ticker']

            items = extractor.execute_extract_process(doc=doc, ticker=ticker)
            items_key = list(items.keys())[0]
            doc_meta = sel2[sel2.Document == doc.Document].to_dict('record')[0]
            for acct, val in items[items_key].items():
                if type(val) == str: continue
                scale = extractor.config[ticker].accounts[acct].scale
                rec = RecordMetadata(
                                cik = str(doc_meta['short_cik']),
                                accn = str(doc_meta['accession_number']),
                                form = f"{doc_meta['file_type']}/{doc.Type}",
                                account = acct,
                                value = scale_value(val, scale),
                                account_title = extractor.config[ticker].accounts[acct].table_account,
                                xbrl_tag = extractor.config[ticker].accounts[acct].xbrl,
                                fy = np.nan,
                                fp = np.nan,
                                end = np.nan,
                                filed = doc_meta['file_date']
                        )
                recs.append(rec)
        #TODO:load raw recs individually to db table
        df = pd.DataFrame(recs, columns=RecordMetadata._fields)
        df.to_csv(intermediate_step, index=False)

    # prepare table
    if df.shape[0] > 0:
        df_columns = df.drop(labels=['account','value'], axis=1).drop_duplicates(subset=['cik','accn','form'])
        #check for duplicates that will cause error: df[df.duplicated(subset=['cik','accn','acct'])==True].shape
        df_wide = df.pivot(index=['cik','accn','form'], columns='account', values='value').reset_index()

        tmp_key = pd.DataFrame(  df.groupby(['cik','accn'])['account'].apply(list) )
        tmp_key['account_title'] = pd.DataFrame(  df.groupby(['cik','accn'])['account_title'].apply(list) )['account_title']
        tmp_key['xbrl_tag'] = pd.DataFrame(  df.groupby(['cik','accn'])['xbrl_tag'].apply(list) )['xbrl_tag']
        tmp_key['titles'] = tmp_key.apply(lambda row: str(list( zip(row['account'], row['account_title'], row['xbrl_tag']))), axis=1 )
        tmp_key['index'] = tmp_key.index
        tmp_key.reset_index(inplace=True)
        df_wide['titles'] = tmp_key['titles']        

        df_wide_total = pd.merge(df_wide, df_columns, on=['cik','accn','form'], how='left')
        df_wide_total['filed'] = pd.to_datetime(df_wide_total['filed'])
        df_wide_total.sort_values(by='filed', ascending=False, inplace=True)
        df_8k = df_wide_total.copy(deep=True)

        # TODO: separate into another function.  df_10q may be empty
        df_10q = pd.read_sql(db.table_name, con=db.engine)
        if df_10q.shape[0] > 0:
            table_cols = df_10q.columns.to_list()
            df_10q['yr-qtr'] = df_10q.apply(create_qtr, axis=1)
            df_10q['short_cik'] = df_10q['cik'].astype(str)
            df_8k['yr-qtr'] = get_8k_qtr(df_8k, df_10q)
            df_8k['fy'] = df_8k['yr-qtr'].str.split(pat='-').str[0]
            df_8k['fp'] = df_8k['yr-qtr'].str.split(pat='-').str[1].replace({'1':'Q1','2':'Q2','3':'Q3','4':'FY'})
            current_cols = df_8k.columns
            select_cols = [col for col in current_cols if col in table_cols]
            df_to_commit = df_8k[select_cols]
        else:
            df_8k['fy'] = None
            df_8k['fp'] = None
            df_to_commit = df_8k
        
    # load into db
        try:
            df_to_commit.to_sql(db.table_name, 
                    con=db.engine,
                    if_exists='append', 
                    index=False
                    )
        except sql.exc.IntegrityError as e:
            logger.warning('Unique key violation on insert')
        else:
            logger.info(f'Inserted {df_8k.shape[0]} records to table {db.table_name}')
        return True
    else:
        return False




def poll_sec_edgar(db, ciks):
    """Check for changes in SEC EDGAR DB, 
    and make updates to sqlite3 when appropriate.  TODO:<<< no, updates are done else where
    """
    today = pd.Timestamp(date.today()).__str__()    #'2022-03-01 00:00:00'
    form_types = ['8-K','10-K','10-Q']

    result_df = []
    for cik in ciks:
        recent = api_request(type='firm_details', cik=cik, acct=None)
        recs = []
        keys = recent.keys()
        for item in range(len(recent['filingDate'])):
            rec = {val:recent[val][item] for val in keys}
            recs.append(rec)

        df = pd.DataFrame.from_dict(recs)
        df['filingDate_t'] = pd.to_datetime(df['filingDate'])
        df_tmp1 = df[df['filingDate_t'] > today]
        df_tmp2 = df_tmp1[df_tmp1['form'].isin(form_types)]
        if df_tmp2.shape[0] > 0:
            df_tmp2['cik'] = cik
            result_df.append(df_tmp2)

    return result_df
    '''
    for df in result_df:
        for form in range(df_tmp2.shape[0]):
            if df_tmp2['form'] == '8-K':
                #download/extract 8-K
                #df_tmp2['accessionNumber'] 
                return True 
            else:
                #update 10-K/10-Q
                return True
    return False
    '''

def create_qtr(row):
    """Create the column formatted `yr-qtr` using pd.apply(<fun>, axis=1)."""
    fy = row['fy']
    fp = row['fp']
    if fy == None or fp == None:
        qtr = row['dt_filed'].quarter
        yr = row['dt_filed'].year
    else:
        yr = fy
        if row['fp']=='FY': 
            qtr = '4'
        else:
            qtr = str(row['fp']).replace('Q','')
    return f'{yr}-{qtr}'




def create_report(report_type, db, output_path):
    """Create the final report from db query."""

    def report_long(output_path=False):
        df = db.query_database()
        df['dt_filed'] = pd.to_datetime(df.filed)
        df.sort_values(by=['cik','dt_filed'], inplace=True, ascending=False)
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
        qtrly = ['10-K','10-Q']
        df8k = df_long[~df_long.form.isin(qtrly) & (pd.isna(df_long['ACL'])==False)]
        df8k['ACL_num'] = df8k['ACL'].apply(lambda x: np.abs( pd.to_numeric(x, errors='coerce')) )
        df8k['Loans_num'] = df8k['Loans'].apply(lambda x: np.abs( pd.to_numeric(x, errors='coerce')) )
        
        now = pd.Timestamp(datetime.now())
        initial_qtr = f'{now.year}-{now.quarter}'
        df8k['yr-qtr'] = pd.DatetimeIndex(df8k['dt_filed']).year.astype(str) + '-Q' + pd.DatetimeIndex(df8k['dt_filed']).quarter.astype(str)
        df8k.sort_values(by='dt_filed', inplace=True)
        
        dfcnt = pd.DataFrame( df8k['yr-qtr'].value_counts() )
        dfcnt['date'] = pd.PeriodIndex(dfcnt.index, freq='Q').to_timestamp()
        dfcnt['qtrs'] = dfcnt.index
        dfcnt.sort_values(by='date', ascending=False, inplace=True)
        qtrs = dfcnt.qtrs.tolist()

        # get df for each qtr
        df_result = pd.DataFrame()
        df_result['cik'] = df_long['cik'].unique()
        df_result.set_index('cik', inplace=True)
        dfs = {}
        for qtr in qtrs:
            df_tmp = df8k[df8k['yr-qtr'] == qtr].drop_duplicates(subset='cik')
            df_tmp.set_index('cik', inplace=True)
            if df_tmp.shape[0] > 0:
                dfs[qtr] = df_tmp

        # create df by adding columns for each qtr
        meta = namedtuple('meta_record', ['accn', 'form', 'titles'])
        df_Meta = df_result.copy(deep=True)
        df_ACL = df_result.copy(deep=True)
        df_Loans = df_result.copy(deep=True)
        df_Ratio = df_result.copy(deep=True)
        for key in dfs.keys():
            if dfs[key].shape[0] > 0:

                recs = []
                for row in dfs[key].to_dict('records'):
                    rec = meta(
                        accn = row['accn'],
                        form = row['form'],
                        titles= row['titles']
                    )
                    recs.append(rec)
                df_Meta['accn'+'|'+key] = recs
                #df_Meta = df_Meta.join(meta, how='outer') 
                #df_Meta.rename(columns={'accn':'accn'+'|'+key}, inplace=True)

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
        firms = [(36104, 'USB'), (4962, 'AXP'), (927628, 'COF'), (35527, 'FITB'), (49196, 'HBAN'), (91576, 'KEY'), (895421, 'MS'), (19617, 'JPM'), (831001, 'C'), (72971, 'WFC'), (70858, 'BAC'), (713676, 'PNC'), (886982, 'GS'), (92230, 'TFC'), (40729, 'ALLY'), (759944, 'CFG'), (1504008, 'BKU')]
        df_ACL['cik'] = df_ACL.index
        bank = df_ACL['cik'].apply(lambda x: [item[1] for item in firms if str(item[0])==x][0] )
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
            url = 'www.sec.gov/Archives/edgar/data/{cik}/{accn_wo_dash}/{accn}-index.htm'
            hdr_rows = 2
            time_periods = len(qtrs)
            col_end = col_start + time_periods
            row_end = row_start + df.shape[0]
            data_row_start = row_start + hdr_rows
            data_row_end = row_end + hdr_rows

            worksheet.set_column(col_start, col_end - 1, 11.5)          #header requires `-1` because it is inclusive
            worksheet.set_row(row_start, 20)
            worksheet.set_row(row_start+1, 20)
            worksheet.merge_range(row_start,col_start, row_start,col_end-1, hdr_title, header_format)
            for idx, qtr in enumerate( qtrs ):
                col = col_start + idx
                worksheet.write_string(1, col, qtr, header_format)

            for idxC, col in enumerate( range(col_start, col_end)):
                for idxR, data_row in enumerate( range(data_row_start, data_row_end)):
                    raw_value = df.iloc[idxR, idxC + df_col_offset]
                    meta_value = {'cik': df_meta.index.tolist()[idxR], 
                                    'accn': df_meta.iloc[idxR, idxC].accn, 
                                    'form': df_meta.iloc[idxR, idxC].form, 
                                    'title': ast.literal_eval(df_meta.iloc[idxR, idxC].titles)[0][1],
                                    'xbrl': ast.literal_eval(df_meta.iloc[idxR, idxC].titles)[0][2]
                                    }
                    if raw_value > 1:
                        value = raw_value / 1000000
                        data_format = workbook.add_format({'num_format': '#,##0.0', 'border':1})
                    else:
                        value = raw_value
                        data_format = workbook.add_format({'num_format': '0.000', 'border':1})
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
        worksheet = workbook.add_worksheet()

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
        missing_format = workbook.add_format({
            'align': 'center',
            'font_color': 'gray'
            })
        comment_format = {
            'visible': False,
            'width': 200,
            'height': 125,
            'color': '#f7f7f5'
            }

        for idx, ticker in enumerate(banks):
            row = idx + 2
            worksheet.write_string(row, 1, ticker, header_format)

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
        df['yr-qtr'] = df['variable'].str.split('|').str[1]
        df['dt_filed'] = pd.PeriodIndex(df['yr-qtr'], freq='Q').to_timestamp()
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
        df_long['yr-qtr'] = df_long.apply(create_qtr, axis=1)

        df_result = pd.DataFrame()
        df_result['cik'] = df_long['cik'].unique()
        df_result.set_index('cik', inplace=True)
        dfs = {}
        for days in days_from_last_qtr:
            prev = pd.Timestamp(datetime.now())  -  pd.to_timedelta( days, unit='d')
            prev_qtr = f'{prev.year}-{prev.quarter}'
            # try 10-K/-Q first
            # mask = ~df_long['form'].isin( qtrly ) if days == 90 else df_long['form'].isin( qtrly )
            # df_long[ (df_long['yr-qtr'] == '2021-4') & (~df_long['form'].isin( qtrly ))]
            mask = df_long['form'].isin( qtrly )
            df_10k_qtr = df_long[ (df_long['yr-qtr'] == prev_qtr) & (pd.isna(df_long['ACL'])==False) & (mask)]
            ciks_10k = df_10k_qtr.cik.unique()
            if len(ciks_10k) < len(ciks):
                mask = ~df_long['form'].isin( qtrly )
                df_8k_qtr = df_long[ (df_long['yr-qtr'] == prev_qtr) & (pd.isna(df_long['ACL'])==False)  & (mask)]
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

        
    dir_path = Path(output_path).parent
    match report_type:
        case 'long': 
            result = report_long(output_path)
        case 'accounting_policy':
            df_long = report_long() 
            result = report_accounting(df_long, dir_path)
        case 'trend':
            df_long = report_long()
            #result = template(df_long, dir_path)
            df_ACL = report_accounting(df_long, dir_path)
            result = trend(df_ACL, dir_path)
        case 'validate':
            df_long = report_long()
            result = validate(df_long, dir_path)

    return result


def reset_files(mode='intermediate_files'):
    """Remove specific files from the `downloades` directory.
    'intermediate_files' - created during extraction process
    'directory_structure' - all files, dirs created during download and extraction process
    """
    match mode:
        case 'intermediate_files':
            dir_path = './archive/downloads/sec-edgar-filings'
            acct_names = ['ACL', 'ALLL', 'PCL', 'Loans', 'ChargeOffs', 'ACLpctLoan', 'Charge Offs', 'ACLpct Loan']

            p = Path(dir_path)
            files = list(p.glob('**/*'))
            files_to_del = []
            for file in files:
                if not hasattr(file, 'stem'): continue
                if file.stem in acct_names:
                    files_to_del.append(file)
            [file.unlink() for file in files_to_del]
            return True
        case 'directory_structure':
            pass