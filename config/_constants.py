#!/usr/bin/env python3
"""
Declare constants and initialize configurations.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData

#builtin
import logging
from pathlib import Path
from collections import namedtuple

#my libs
from sec_edgar_extractor.extract import Extractor
from sec_edgar_downloader import UrlComponent as uc





# file managmeent
FILE_EMAILS = './config/emails.csv'
FILE_FIRMS = './config/ciks_test.csv'

DIR_REPORTS = './archive/report'
DIR_SEC_DOWNLOADS = './archive/downloads'
FILE_LOG = './archive/process.log'
FILE_DB = './archive/prod.db'
#FILE_DB = f'sqlite:///{db_file}'     #for in-memory testing: 'sqlite://'
EMAIL_NETWORK_DRIVE = r'\\hqfiles01\sec_edgar$\sec_edgar'


# request management
"""
SEC limits users to no more than 10 requests per second
Sleep 0.1s between each request to prevent rate-limiting
Source: https://www.sec.gov/developer
"""
SEC_EDGAR_RATE_LIMIT_SLEEP_INTERVAL = 0.1

# Number of times to retry a request to sec.gov
MAX_RETRIES = 5

# Time to wait for response from server (seconds)
DEFAULT_TIMEOUT = 5

MINUTES_BETWEEN_CHECKS = 0.10
QUARTERS_IN_TABLE = 6




# database schema
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



# initialize logging
class Logger:

    def __init__(self, log_file):
        self.log_file = Path(log_file)

    def create_logger(self):
        """Create logger and associated file (if necessary)."""
        if not self.log_file.parent.is_dir():
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.is_file():
            open(self.log_file, 'x').close()

        logging.basicConfig(filename=self.log_file, 
                            encoding='utf-8', 
                            level=logging.INFO, 
                            format='%(asctime)s %(message)s')
        logger = logging.getLogger(__name__)
        return logger

logger = Logger(FILE_LOG).create_logger()




# initialize firms
def load_firms(file_path):
    """Load file containing firms."""
    df = pd.read_csv(file_path)
    firm_recs = df.to_dict('records')
    firms = []
    for firm in firm_recs:
        #if firm['Scope']=='In':
        item = uc.Firm(ticker = firm['Ticker'] )
        item.Scope = firm['Scope']
        firms.append( item )
    return firms

LIST_ALL_FIRMS = load_firms(FILE_FIRMS)



# initialize extractor
extractor = Extractor(save_intermediate_files=True)
extractor.config.load_config_from_file()
accounts_domain = []
[accounts_domain.extend(item.accounts.keys()) 
    for item in extractor.config.get(report_date=None, mode='firm_records_no_filter').values()]
LIST_ALL_ACCOUNTS = list(set(accounts_domain))



# initialize schema
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
    *(Column(acct, Integer()) for acct in LIST_ALL_ACCOUNTS),
    Column('titles', String),
    Column('fy', String),
    Column('fp', String),
    Column('end', String),
    Column('filed', String),
    Column('yr_qtr', String)
    )
schema_filing = [col.name for col in filings.columns if col.name not in LIST_ALL_ACCOUNTS]
schema_filing.extend(['acct', 'val'])       
FilingMetadata = namedtuple("FilingMetadata", schema_filing) 

LIST_ALL_TABLES = [
                {'name': 'records', 'table': records, 'schema': schema_records}, 
                {'name': 'filings', 'table': filings, 'schema': schema_filing}
                ]