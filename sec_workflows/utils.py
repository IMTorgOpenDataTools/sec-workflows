#!/usr/bin/env python3
"""
Module Docstring
"""
from asyncio.log import logger
import pandas as pd
import sqlalchemy as sql
from pathlib import Path
import time
from datetime import date, timedelta
import requests

from sec_edgar_downloader import Downloader, Firm

import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    log_file,
    sec_edgar_downloads_path,
    db_file,
    table_name,
    MINUTES_BETWEEN_CHECKS,
    QUARTERS_IN_TABLE,
    accts,
    meta,
    filings,
    FilingMetadata
)

url_firm_details = "https://data.sec.gov/submissions/CIK{}.json"
url_api_account = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'     #long_cik, acct



def load_firms(file_path):
    """Load file containing firms."""
    df = pd.read_csv(file_path)
    firm_lst = df['CIK'].to_list()
    cik_lst = [str(cik).zfill(10) for cik in firm_lst]
    ticker_lst = df['Ticker'].to_list()
    return cik_lst, ticker_lst


 


def api_request(type, cik, acct):
    headers = {
        'User-Agent': 'RandomFirm',
        'From': 'first.last@gmail.com',
        'Accept-Encoding': 'gzip, deflate'
        }
    url_firm_details = "https://data.sec.gov/submissions/CIK{}.json"
    url_company_concept = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'     #long_cik, acct
    match type:
        case 'firm_details': 
            filled_url = url_firm_details.format(cik)
            keys = ('filings','recent')
        case 'concept': 
            filled_url = url_company_concept.format(cik, acct)
            keys = ('units','USD')
        case _:
            logger.warning(f'`api_request` type={type} is not an option (`firm_detail`, `concept`)')
            exit()
    edgar_resp = requests.get(filled_url, headers=headers)
    if edgar_resp.status_code == 200:
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
    banks = [Firm(ticker=bank[1]) for bank in tickers]
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






def initialize_db(db, ciks, accts):
    """Initialize the database
    """
    #TODO: use requests.session to improve speed
    recs = []
    for cik in ciks:
        for acct in accts.values():
            logger.info(f'api request for account: {acct}')
            items = api_request(type='concept', cik=cik, acct=acct)
            if not items:
                break
            items.reverse()
            items_dedup = remove_list_dups(items, 'accn')
            items_dedup.reverse()
            end = len(items_dedup)
            idx_start = end - QUARTERS_IN_TABLE if QUARTERS_IN_TABLE < end  else end 
            tgt_items = items_dedup[idx_start:]
            for item in tgt_items:
                rec = FilingMetadata(
                        cik = cik,
                        accn = item['accn'],
                        acct = acct,
                        val = item['val'],
                        fy = item['fy'],
                        fp = item['fp'],
                        form = item['form'],
                        end = item['end'],
                        filed = item['filed']
                )
                recs.append( rec )
            time.sleep(1)
    if len(recs) > 0:
        df = pd.DataFrame(recs, columns=FilingMetadata._fields)
        df_columns = df.drop(labels=['acct','val'], axis=1).drop_duplicates(subset=['cik','accn'])
        df_wide = df.pivot(index=['cik','accn'], columns='acct', values='val').reset_index()
        df_wide_total = pd.merge(df_wide, df_columns, on=['cik','accn'], how='left')
        df_wide_total['filed'] = pd.to_datetime(df_wide_total['filed'])
        df_wide_total.sort_values(by='filed', ascending=False, inplace=True)
        try:
            df_wide_total.to_sql(db.table_name, 
                    con=db.engine,
                    if_exists='append', 
                    index=False
                    )
        except sql.exc.IntegrityError as e:
            logger.warning('Unique key violation on insert')
            return False
        else:
            logger.info(f'Inserted {df_wide_total.shape[0]} records to table {db.table_name}')
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



def create_report(report_type, db, output_path):
    """Create the final report from db query."""

    def report_long(output_path):
        df = db.query_database()
        if df.shape[0] > 0:
            df.to_csv(output_path, index=False)
            logger.info(f'Report saved to path: {output_path}')
        else:
            logger.info(f'No data available to report')
        return None


    match report_type:
        case 'long': report_long(output_path)
        case 'wide': pass

    return None
