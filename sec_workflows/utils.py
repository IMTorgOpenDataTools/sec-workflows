#!/usr/bin/env python3
"""
Module Docstring
"""
from asyncio.log import logger
from collections import namedtuple
import subprocess
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
    emails_file,
    firms,
    sec_edgar_downloads_path,
    MAX_RETRIES,
    SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL,
    DEFAULT_TIMEOUT,
    MINUTES_BETWEEN_CHECKS,
    QUARTERS_IN_TABLE,
    accts,
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








def send_notification():
    """Send notification that report is updated."""

    subject = 'SEC Report Update'
    body = "Dear Sir/Ma'am, this is a notification that the SEC earnings report is updated.  You can find it in the following shared drive: `\hqfile01\sec`."

    df_emails = pd.read_csv(emails_file)
    emails = df_emails['address'].tolist()

    for email in emails:
        test = subprocess.Popen(["mailx", "-s", subject, email, "<", body], stdout=subprocess.PIPE)
        output = test.communicate()[0]

    return True



 
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
'''
def get_recent_financial_release(tickers, ciks):
    """Search and download the most-recent 8-K/EX-99.* documents."""
    def check(row): 
        return "EX-99.1" in row.Type if type(row.Type) == str else False   #TODO:add 99.2+

    start = date.today() - timedelta(days=90)
    banks = [uc.Firm(ticker=bank[1]) for bank in tickers]
    dl = Downloader(sec_edgar_downloads_path)
    for bank in banks:
        ticker = bank.get_info()["ticker"]
        urls = dl.get_metadata("8-K",
                            ticker,
                            after = start.strftime("%Y-%m-%d")
                            )
    df = dl.filing_storage.get_dataframe(mode="document")
    sel1 = df[(df["short_cik"].isin(ciks)) & (df["file_type"] == "8-K") & (df["FS_Location"] == "")]
    mask = sel1.apply(check, axis=1)
    sel2 = sel1[mask] 
    lst_of_idx = sel2.index.tolist()
    staged = dl.filing_storage.get_document_in_record( lst_of_idx )
    downloaded_docs = dl.get_documents_from_url_list(staged)
    return downloaded_docs
'''

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


def create_qtr(row):
    """Create the column formatted `yr-qtr` using pd.apply(<fun>, axis=1).
    This is based on columns `end`-of-period data, not date-`filed`, which may be
    more than two months, later.
    
    """
    if row['end'] == None:
        dt = pd.to_datetime(row['filed'])
    else:
        dt = pd.to_datetime(row['end'])
    fy = row['fy']
    fp = row['fp']
    if fy == None or fp == None:
        qtr = dt.quarter
        yr = dt.year
    else:
        yr = fy
        if row['fp']=='FY': 
            qtr = 'Q4'
        else:
            qtr = str(row['fp'])
    return f'{yr}-{qtr}'





def poll_sec_edgar(db, firms, after_date):
    """Check for changes in SEC EDGAR DB, and return corresponding firms"""

    after = pd.Timestamp(after_date).__str__()    #'2022-03-01 00:00:00'

    client = requests.Session()
    client.mount("http://", HTTPAdapter(max_retries=retries))
    client.mount("https://", HTTPAdapter(max_retries=retries))

    form_updates = {'8k':set(), 
                    '10kq':set()
                    }
    for firm in firms:
        cik = firm._cik
        if cik:
            recent = api_request(session=client, type='firm_details', cik=cik, acct=None)
            recs = []
            keys = recent.keys()
            for item in range(len(recent['filingDate'])):
                rec = {val:recent[val][item] for val in keys}
                recs.append(rec)

            df = pd.DataFrame.from_dict(recs)
            df['filingDate_t'] = pd.to_datetime(df['filingDate'])
            df_tmp1 = df[df['filingDate_t'] > after]                #ensure `after` is changed for testing
            if df_tmp1.shape[0] > 0:
                for row in df_tmp1.to_dict('records'):
                    if row['form'] in ['8-K']:
                        form_updates['8k'].add(firm)
                    if row['form'] in ['10-K', '10-Q']:
                        form_updates['10kq'].add(firm)
    return form_updates



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