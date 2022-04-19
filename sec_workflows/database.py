#!/usr/bin/env python3
"""
Module Docstring
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"



#third-party
import pandas as pd
import sqlalchemy as sql                     #create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData
import sqlalchemy_utils as sql_util         #database_exists, create_database

#built-in
from pathlib import Path
import logging
import time
from datetime import date
import requests

#my libs
from utils import (
    remove_list_dups,
)
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    db_file,
    table_name,
    MINUTES_BETWEEN_CHECKS,
    QUARTERS_IN_TABLE,
    accts,
    meta,
    filings,
    FilingMetadata
)




class Database:
    """Database class."""

    def __init__(self, db_file, table_name, meta, logger):
        self.db_file = db_file
        self.db_path = f'sqlite:///{db_file}' 
        self.table_name = table_name
        self.meta = meta
        self.engine = None
        self.logger = logger


    def check_db_file(self):
        path = Path(self.db_file)
        if not path.is_file():
            self.logger.warning('Database file not found.')
            return False
        else:
            self.logger.warning('Database file exists.')
            return True


    def check_db_schema(self):
        engine = sql.create_engine(self.db_path, echo=True)
        self.engine = engine
        self.meta.reflect(bind=engine)
        if not sql_util.database_exists(engine.url) and not sql.inspect(engine).has_table(self.table_name):      #not engine.dialect.has_table(engine, table_name, schema):
            self.logger.warning('Database does not exist.  Making it now.')     
            self.meta.create_all(engine)                                                                         #create_database(engine.url)
            self.logger.warning('Database created.') 
        else:
            check = self.validate_db()
            if check:
                self.logger.info('Database table and schema correct.')
        return None


    def validate_db(self):
        """Validate the database table `filings` is populated."""
        #filings: not sql_util.database_exists(engine.url) and not sql.inspect(engine).has_table(table_name), get_columns(tbl_name),
        return True


    def insert_df_to_table(self):
        pass


    def update_database(self):
        pass


    def query_database(self):
        '''
        meta.reflect(bind=engine)
        filings = meta.tables[table_name]
        s = filings.select()
        result = engine.execute(s)
        for row in result:
            print (row)
        '''
        df = pd.read_sql_table(self.table_name, self.engine)
        return df