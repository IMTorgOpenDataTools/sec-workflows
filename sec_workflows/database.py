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
#from utils import (remove_list_dups,)
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    db_file,
    tables_list,
    MINUTES_BETWEEN_CHECKS,
    QUARTERS_IN_TABLE,
    accts,
    meta,
    filings,
    FilingMetadata
)




class Database:
    """Database class."""

    def __init__(self, db_file, tables_list, meta, logger):
        self.db_file = db_file
        self.db_path = f'sqlite:///{db_file}' 
        self.table_name = tables_list
        self.meta = meta
        self.engine = None
        self.logger = logger


    def check_db_file(self):
        """Check the database file exists."""
        path = Path(self.db_file)
        if not path.is_file():
            self.logger.warning('Database file not found.')
            #TODO:create db file
            return False
        else:
            self.logger.warning('Database file exists.')
            return True


    def check_db_schema(self):
        """Check the database table schemas are correct."""
        engine = sql.create_engine(self.db_path, echo=True)
        self.engine = engine
        self.meta.reflect(bind=engine)
        if not sql_util.database_exists(engine.url):
            check_tables = []
            for tbl in self.table_name:
                check = sql.inspect(engine).has_table(tbl['name']) 
                check_tables.append(check)
                if not all(check_tables):
                    self.logger.warning('Database does not exist.  Making it now.')     
                    self.meta.create_all(engine)                                                                         
                    self.logger.warning('Database created.') 
        else:
            self.logger.info('Database table and schema correct.')
            check = self.validate_db_records()
            if check:
                self.logger.info('Database table are populated.')
        return None


    def validate_db_records(self):
        """Validate the database tables are populated."""
        populated = []
        for tbl in self.table_name:
            stmt = sql.select( tbl['table'] )
            with sql.orm.Session(self.engine) as session:
                row = session.execute(stmt).first()
                populated.append( row )
        if all(populated):
            return True
        else:
            return False


    def initialize_db(self):
        """Populate database with initial import of data from
        multiple data sources."""
        pass


    def populate_quarterly_filings(self):
        """Import quarterly (10-K/-Q) data from SEC EDGAR API."""
        pass


    def populate_earnings_releases(self):
        """Download, extract, and import earnings (8-K_ data 
        from SEC EDGAR API."""
        pass


    def populate_firm_data(self):
        """Download and import firm data."""
        pass


    def populate_xbrl_descriptions(self):
        """Scrape and import XBRL description data."""
        pass


    def get_filing_df(self):
        """Get dataframe of filings from database table."""
        pass



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
        df = pd.read_sql_table(self.table_name[1]['name'], self.engine)
        return df