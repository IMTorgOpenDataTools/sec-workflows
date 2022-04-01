#!/usr/bin/env python3
"""
Module Docstring
"""
from asyncio.log import logger
import pandas as pd
from pathlib import Path
import time
from datetime import date
import requests
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    log_file,
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



def load_target_firms():
    """Load file containing firms."""
    pass


#TODO: def request_from_api():
headers = {
        'User-Agent': 'IMTorg',
        'From': 'jason.beach@mgmt-tech.org',
        'Accept-Encoding': 'gzip, deflate'
        }


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




def initialize_db(db, ciks, accts):
    """Initialize the database
    """
    #TODO: use requests.session to improve speed
    recs = []
    for cik in ciks:
        for acct in accts.keys():
            print(acct)
            url_filled = url_api_account.format(cik, acct) 
            edgar_resp = requests.get(url_filled, headers=headers)
            items = edgar_resp.json()['units']['USD']
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
                        acct = accts[acct],
                        val = item['val'],
                        fy = item['fy'],
                        fp = item['fp'],
                        form = item['form'],
                        end = item['end'],
                        filed = item['filed']
                )
                recs.append( rec )
            time.sleep(1)
    df = pd.DataFrame(recs, columns=FilingMetadata._fields)

    df_columns = df.drop(labels=['acct','val'], axis=1).drop_duplicates(subset=['cik','accn'])
    df_wide = df.pivot(index=['cik','accn'], columns='acct', values='val').reset_index()
    df_wide_total = pd.merge(df_wide, df_columns, on=['cik','accn'], how='left')
    df_wide_total['filed'] = pd.to_datetime(df_wide_total['filed'])
    df_wide_total.sort_values(by='filed', ascending=False, inplace=True)
    df_wide_total.to_sql(db.table_name, 
                con=db.engine,
                if_exists='append', 
                index=False
                )
    #print(df_wide_total)
    print(f'Inserted {df_wide_total.shape[0]} records to table {db.table_name}')




def poll_sec_edgar(db, ciks):
    """Check for changes in SEC EDGAR DB, 
    and make updates to sqlite3 when appropriate.  TODO:<<< no, updates are done else where
    """
    today = pd.Timestamp(date.today()).__str__()    #'2022-03-01 00:00:00'
    form_types = ['8-K','10-K','10-Q']

    result_df = []
    for cik in ciks:
        url_filled = url_firm_details.format(cik)
        edgar_resp = requests.get(url_filled, headers=headers)
        recent = edgar_resp.json()['filings']['recent']

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
        df.to_csv(output_path, index=False)
        logger.info(f'Report saved to path: {output_path}')
        return None


    match report_type:
        case 'long': report_long(output_path)
        case 'wide': pass

    return None
