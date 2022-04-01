#!/usr/bin/env python3
"""
Module Docstring
"""

from collections import namedtuple
from sqlalchemy import Table, Column, Integer, String, MetaData


log_file = './archive/example.log'

db_file = './archive/test.db'
#db_path = f'sqlite:///{db_file}'     #for testing: 'sqlite://'
table_name = 'filings'

MINUTES_BETWEEN_CHECKS = 1
QUARTERS_IN_TABLE = 6
OUTPUT_REPORT_PATH = './archive/output.csv'

accts = {'NotesReceivableGross': 'Total_Loans'}

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