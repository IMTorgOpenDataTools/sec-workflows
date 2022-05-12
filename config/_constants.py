#!/usr/bin/env python3
"""
Module Docstring
"""
import math
from collections import namedtuple
from sqlalchemy import Table, Column, Integer, String, MetaData
import pandas as pd

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



'''
def load_accounts(file_path):
    """Load file containing accounts."""
    account_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        k,v = line.replace('\n','').split(',')
        account_dict[k] = v
    return account_dict  


account_defaults = {
    'ACL': {'xbrl':'FinancingReceivableAllowanceForCreditLosses','term':'Allowance for credit loss'},
#   'ALLL': {'xbrl':'FinancingReceivableAllowanceForCreditLosses','term':'allowance for loan and lease losses'},
#   'PCL': {'xbrl':'ProvisionForLoanLeaseAndOtherLosses','term':'Provision for credit loss'},
#   'ChargeOffs': {'xbrl':'FinancingReceivableAllowanceForCreditLossesWriteOffsNet','term':'charge-off'},
    'Loans': {'xbrl':'NotesReceivableGross','term':'Loan'},
#    'ACLpctLoan': {'xbrl':'FinancingReceivableAllowanceForCreditLossToOutstandingPercent','term':'as percent of loans'},
#    'ALLpctLoan': {'xbrl':'FinancingReceivableAllowanceForCreditLossToOutstandingPercent','term':'as percent of loans'},
#    'ALLLpctLHFI': {'xbrl':'FinancingReceivableAllowanceForCreditLossToOutstandingPercent','term':'as percent of LHFI'}
}
'''

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

'''
def load_config_account_info(file=None):
    """"Load all account_info and return config(uration) dict.
    TODO:add more defaults 
    """
    def get_default_if_missing(rec, key):
        defaults = account_defaults
        acct = account
        return rec[key] if math.isnan(rec[key]) == False else defaults[acct]["term"]

    if file==None:
        file = "./config/Firm_Account_Info.csv"
    df = pd.read_csv(file, na_values=["NA",""])
    tickers = df["ticker"].value_counts().index
    accounts = df["name"].value_counts().index
    config = {}

    for ticker in tickers:
        tmp_accts = {}
        for account in accounts:
            tmp_df = df[(df["ticker"]== ticker) & (df["name"]==account)]
            if tmp_df.shape[0] == 1:
                tmp_rec = tmp_df.to_dict("records")[0]
                tmp_acct = AccountRecord(                                
                    name = tmp_rec["name"],
                    xbrl = tmp_rec["xbrl"],
                    table_name = tmp_rec["table_name"],
                    table_account = tmp_rec["table_title"],
                    table_column = tmp_rec["col_idx"],
                    scale = tmp_rec["scale"],
                    discover_terms = get_default_if_missing(rec=tmp_rec, key="discover_terms"),
                    search_terms = get_default_if_missing(rec=tmp_rec, key="search_terms"),
                    exhibits = tmp_rec["exhibits"]
                )
                tmp_accts[account] = tmp_acct
            else:
                #print(f'ERROR: tmp_df has {tmp_df.shape[0]} rows')
                break
        tmp_firm = FirmRecord(
                Firm = ticker,
                accounts = tmp_accts
                )
        config[ticker] = tmp_firm

    return config
'''

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