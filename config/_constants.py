#!/usr/bin/env python3
"""
Module Docstring
"""
import math
from collections import namedtuple
import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData
#from sqlalchemy.orm import declarative_base
#from sqlalchemy.orm import relationship

#Base = declarative_base()


from sec_edgar_extractor.extract import Extractor
from sec_edgar_downloader import UrlComponent as uc


# SEC limits users to no more than 10 requests per second
# Sleep 0.1s between each request to prevent rate-limiting
# Source: https://www.sec.gov/developer
SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL = 0.1

# Number of times to retry a request to sec.gov
MAX_RETRIES = 5

# Time to wait for response from server (seconds)
DEFAULT_TIMEOUT = 5

MINUTES_BETWEEN_CHECKS = 0.10
QUARTERS_IN_TABLE = 6
OUTPUT_REPORT_PATH = './archive/report/long_output.csv'



AccountRecord = namedtuple(
    "AccountRecord",
    [   'name',
        'xbrl',
        'table_name',
        'table_account',
        'table_column',
        'scale',
        'discover_terms',
        'search_terms',
        'exhibits'
    ]
)

FirmRecord = namedtuple(
    "FirmRecord",
    [   'Firm',
        'accounts'
    ]
)


def load_firms(file_path):
    """Load file containing firms."""
    df = pd.read_csv(file_path)
    firm_recs = df.to_dict('records')
    firms = []
    for firm in firm_recs:
        if firm:
            item = uc.Firm(ticker = firm['Ticker'] )
            item.Scope = firm['Scope']
            firms.append( item )
    return firms




firms_file = './config/ciks_test.csv'
#accounts_file = './config/accounts.csv'
#accounts_file = './config/Firm_Account_Info.csv'

sec_edgar_downloads_path = './archive'
log_file = './archive/process.log'
db_file = './archive/test.db'
#db_path = f'sqlite:///{db_file}'     #for testing: 'sqlite://'
emails_file = './config/emails.csv'


firms = load_firms(firms_file)

#accts = load_accounts(accounts_file)    #accts = {'NotesReceivableGross': 'Total_Loans'}"
#config = load_config_account_info(file=accounts_file)
extractor = Extractor(save_intermediate_files=True)
config = extractor.config
tmp = []
[tmp.extend(item.accounts.keys()) for item in config.values()]
accts = list(set(tmp))


meta = MetaData()

records = Table(
    'records', meta, 
    Column('cik', String, primary_key = True), 
    Column('accn', String, primary_key = True), 
    Column('form', String, primary_key = True),
    Column('account', String, primary_key = True),
    Column('value', Integer),
    Column('account_title', String),
    Column('xbrl_tag', String),
    Column('fy', String),
    Column('fp', String),
    Column('end', String),
    Column('filed', String),
    )

schema_records = [col.name for col in records.columns]      
RecordMetadata = namedtuple("RecordMetadata", schema_records) 

filings = Table(
    'filings', meta, 
    Column('cik', String, primary_key = True), 
    Column('accn', String, primary_key = True), 
    Column('form', String, primary_key = True),
    *(Column(acct, Integer()) for acct in accts ),
    Column('titles', String),
    Column('fy', String),
    Column('fp', String),
    Column('end', String),
    Column('filed', String),
    Column('yr_qtr', String)
    )
schema_filing = [col.name for col in filings.columns if col.name not in accts]
schema_filing.extend(['acct', 'val'])       
FilingMetadata = namedtuple("FilingMetadata", schema_filing) 

tables_list = [
                {'name': 'records', 'table': records, 'schema': schema_records}, 
                {'name': 'filings', 'table': filings, 'schema': schema_filing}
                ]