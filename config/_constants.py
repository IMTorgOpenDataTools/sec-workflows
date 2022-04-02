#!/usr/bin/env python3
"""
Module Docstring
"""

from collections import namedtuple
from sqlalchemy import Table, Column, Integer, String, MetaData



def load_accounts(file_path):
    """Load file containing accounts."""
    account_dict = {}
    with open(file_path, 'r') as f:
        for line in f.readlines()[:-1]:
            k,v = line.split(',')
            account_dict[k] = v
    return account_dict  








firms_file = './config/ciks.csv'
accounts_file = './config/accounts.csv'

sec_edgar_downloads_path = './archive'
log_file = './archive/example.log'
db_file = './archive/test.db'
#db_path = f'sqlite:///{db_file}'     #for testing: 'sqlite://'
table_name = 'filings'

MINUTES_BETWEEN_CHECKS = 0.10
QUARTERS_IN_TABLE = 6
OUTPUT_REPORT_PATH = './archive/output.csv'

accts = load_accounts(accounts_file)    #accts = {'NotesReceivableGross': 'Total_Loans'}

meta = MetaData()
filings = Table(
    table_name, meta, 
    Column('cik', String), 
    Column('accn', String, primary_key = True), 
    *(Column(acct, Integer()) for acct in accts.values() ),
    Column('fy', String),
    Column('fp', String),
    Column('form', String),
    Column('end', String),
    Column('filed', String),
    )

schema = [col.name for col in filings.columns if col.name not in list(accts.values())]
schema.extend(['acct', 'val'])       
FilingMetadata = namedtuple("FilingMetadata", schema) 