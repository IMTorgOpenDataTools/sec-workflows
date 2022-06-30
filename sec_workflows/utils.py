#!/usr/bin/env python3
"""
Convenience functions used throughout application.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
import pandas as pd

#builtin
import shutil
from collections import namedtuple
import subprocess
from subprocess import PIPE, STDOUT
from enum import unique

from pathlib import Path
#from datetime import datetime, date, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#libs
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    MAX_RETRIES,
    DEFAULT_TIMEOUT,
    logger
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
            logger.error(f'`api_request` type={type} is not an option (`firm_detail`, `concept`)')
            exit()

    #make request
    edgar_resp = session.get(filled_url, headers=headers)
    if edgar_resp.status_code == 200:
        if keys[0] in edgar_resp.json().keys():
            if keys[1] in edgar_resp.json()[keys[0]].keys():
                items = edgar_resp.json()[keys[0]][keys[1]]
                return items
    else:
        logger.error(f'edgar response returned status code {edgar_resp.status_code}')
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
    """Check for changes in SEC EDGAR DB, and return corresponding firms.
    Only checks if 8k form, does not check if it is an earnings call.
    
    """
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
            df_tmp1 = df[df['filingDate_t'] > after]                
            if df_tmp1.shape[0] > 0:
                for row in df_tmp1.to_dict('records'):
                    if row['form'] in ['8-K']:
                        form_updates['8k'].add(firm)
                    if row['form'] in ['10-K', '10-Q']:
                        form_updates['10kq'].add(firm)
    return form_updates



def delete_folder(pth) :
    for sub in pth.iterdir() :
        if sub.is_dir() :
            delete_folder(sub)
        else :
            sub.unlink()
    shutil.rmtree(pth)



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