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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http

from sec_edgar_downloader import Downloader
from sec_edgar_downloader import UrlComponent as uc
from sec_edgar_extractor.extract import Extractor

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
    config,
    meta,
    filings,
    FilingMetadata
)

url_firm_details = "https://data.sec.gov/submissions/CIK{}.json"
url_api_account = 'https://data.sec.gov/api/xbrl/companyconcept/CIK{}/us-gaap/{}.json'     #long_cik, acct

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
    """Make a request with a given session."""
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





def initialize_db(db, firms):
    """Initialize the database.
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
            logger.info(f'get accounts for firm: {firm._name}')
            accts = extractor.config[ firms[0]._ticker ].accounts.items()
            for acct_key, acct_val in accts:
                logger.info(f'api request for account: {acct_key}')
                items = api_request(session=client, type='concept', cik=firm._cik, acct=acct_val.xbrl)      # if custom xbrl acct fails, then try default
                if not items:
                    continue
                items.reverse()
                items_dedup = remove_list_dups(items, 'accn')
                items_dedup.reverse()
                end = len(items_dedup)
                idx_start = end - QUARTERS_IN_TABLE if QUARTERS_IN_TABLE < end  else end 
                tgt_items = items_dedup[idx_start:]
                for item in tgt_items:
                    rec = FilingMetadata(
                            cik = firm._cik,
                            accn = item['accn'],
                            acct = acct_key,
                            val = item['val'],
                            fy = item['fy'],
                            fp = item['fp'],
                            form = item['form'],
                            end = item['end'],
                            filed = item['filed']
                    )
                    recs.append( rec )
                time.sleep(SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL)

    #prepare records and insert into db
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
