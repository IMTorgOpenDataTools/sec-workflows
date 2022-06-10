#!/usr/bin/env python3
"""
Database class for connecting and managing sqlite3 db.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
from flask import after_this_request
import pandas as pd
import numpy as np
import sqlalchemy as sql                     #create_engine
import sqlalchemy_utils as sql_util         #database_exists, create_database

from sec_edgar_downloader import Downloader
from sec_edgar_downloader import UrlComponent as uc
from sec_edgar_extractor.extract import Extractor
from sec_edgar_extractor.instance_doc import Instance_Doc

#built-in
from asyncio.log import logger
from pathlib import Path
from datetime import date, timedelta

#my libs
import sys
sys.path.append(Path('config').absolute().as_posix() )
sys.path.append(Path('sec_workflows').absolute().as_posix() )
from utils import (
    create_qtr,
    scale_value
)
from _constants import (
    LIST_ALL_ACCOUNTS,
    RecordMetadata,
    FilingMetadata
)




class Database:
    """Database class."""

    def __init__(self, db_file, tables_list, meta, logger, path_download):
        self.db_file = Path(db_file)
        self.db_path = f'sqlite:///{db_file}' 
        self.table_name = tables_list
        self.meta = meta
        self.engine = None
        self.logger = logger
        self.path_download = Path(path_download)

        #db file
        check_file = self.check_db_file()
        if check_file:
            pass
        else:
            self.logger.warning('Fail db creation.')

        #downloads
        if self.path_download.is_dir():
            pass
        else:
            try:
                self.path_download.mkdir(parents=True, exist_ok=True)
            except:
                self.logger.warn('Failed to make downloads path')

        self.downloader = Downloader(self.path_download)
        self.extractor = Extractor(save_intermediate_files=True)
        self.instance_doc = Instance_Doc()


    def check_db_file(self):
        """Check the database file exists."""
        path = self.db_file
        try:
            if not path.is_file():
                self.logger.warning('Database file not found.  Creating now.')
                open(path, 'x').close()
                return True
            else:
                self.logger.warning('Database file exists.')
                return True
        except:
            self.logger.warning('Path does not exist.')
            return False


    def check_db_schema(self):
        """Check the database table schemas are correct."""
        engine = sql.create_engine(self.db_path, echo=True)
        self.engine = engine
        self.meta.reflect(bind=engine)
        try:
            if not sql_util.database_exists(engine.url):
                check_tables = []
                for tbl in self.table_name:
                    check = sql.inspect(engine).has_table(tbl['name']) 
                    check_tables.append(check)
                    if not all(check_tables):
                        self.logger.warning('Database tables do not exist.  Making it them now.')     
                        self.meta.create_all(engine)                                                                         
                        self.logger.warning('Database tables created.') 
            else:
                self.logger.info('Database table and schema correct.')
                check = self.validate_db_records()
                if check:
                    self.logger.info('Database table are populated.')
            return True
        except:
            self.logger.warning('The `check_db_schema` failed.')
            return False



    def validate_db_records(self):
        """Validate the database tables are populated."""
        populated = []
        for tbl in self.table_name:
            stmt = sql.select( tbl['table'] )
            with sql.orm.Session(self.engine) as session:
                try:
                    row = session.execute(stmt).first()
                    populated.append( row )
                except:
                    return False
        if all(populated):
            return True
        else:
            return False


    def get_quarterly_statements(self, firms, after):
        """Get certified quarterly earnings (10-K/-Q) for list of firms, after a date.

        Takes the following steps:
        * discover certified quarterly earnings (10-K/-Q) filings for a list of firms (after a date)
        * determines if they were previously downloaded, continue if not
        * download the instance document
        * extract the xbrl information
        * load into database table `records`
        """
        # configure
        FILE_TYPES = ['10-K', '10-Q']
        AFTER = after                           #"2022-01-01"

        # check if table is initialized
        stmt = "SELECT * FROM " + self.table_name[0]["name"] + " WHERE form like '10-%'"
        df_10q = pd.read_sql(stmt, self.engine)
        #if df_10q.shape[0] > 0: return True
        
        # check if new filing records exist
        ciks = [str(firm._cik) for firm in firms]
        tickers = [(str(firm._cik), str(firm._ticker)) for firm in firms]
        url_new = []
        for firm in firms:
            for filing in FILE_TYPES:
                TICKER = firm.get_info()['ticker']
                url = self.downloader.get_metadata(filing,
                                    TICKER, 
                                    after=AFTER
                                    )
                url_new.extend(url['new'])
        #if not len(url_new) > 0: return True

        # check if new filings are in database, get them if not
        df_fs = self.downloader.filing_storage.get_dataframe(mode='document')
        sel1 = df_fs[(df_fs['short_cik'].isin(ciks)) 
                & (df_fs['file_type'].isin(FILE_TYPES))
                & (df_fs['Type'] == 'XML')
                & (df_fs['Document'].str.contains('_htm.xml', case=True))
                & (pd.to_datetime(df_fs['file_date']) > AFTER)
                ]
        lst_of_idx = sel1.index.tolist()
        staged = self.downloader.filing_storage.get_document_in_record( lst_of_idx )
        updated_docs = self.downloader.get_documents_from_url_list(staged)

        accns = [doc[0].split('|')[1] for doc in updated_docs['new']]
        df_new_filings = df_10q[df_10q['accn'].isin(accns)]
        if set( df_new_filings['accn'].to_list() ) == set( accns ):
            return True
        else:

            # prepare and load the new data items
            selected_docs = list(updated_docs['new']) + list(updated_docs['previous'])
            recs = []
            for key, doc in selected_docs:
                #prepare
                if not doc.FS_Location: continue
                cik = doc.FS_Location.parent.parent.name
                ticker = [item[1] for item in tickers if item[0] == cik][0]
                accn = doc.FS_Location.parent.name
                key = f'{cik}|{accn}'
                filing = self.downloader.filing_storage.get_record(key)
                accts = self.extractor.config[ ticker ].accounts.items()

                #extract xbrl values
                start = pd.to_datetime( filing.file_date )
                with open(doc.FS_Location, 'r') as f:
                    file_htm_xml = f.read()
                df_doc, df = self.instance_doc.create_xbrl_dataframe( file_htm_xml )
                fy = df_doc.value[df_doc['name']=='DocumentFiscalYearFocus'].values[0]
                fp = df_doc.value[df_doc['name']=='DocumentFiscalPeriodFocus'].values[0]
                end = df_doc.value[df_doc['name']=='DocumentPeriodEndDate'].values[0]
                for acct_key, acct_rec in accts:
                    item = rec = None
                    try:
                        xbrl_tag = acct_rec.xbrl                        #<<< TODO:try a list of similar xbrls until one hits
                        subset = df[(df['concept'] == xbrl_tag )               
                                    & (df['dimension'] == '' ) 
                                    & (df['value_context']== '' )
                                    & (df['start'] < start )
                                    & (pd.isna(df['end']) == True )
                                    ]
                        if subset.shape[0] > 0:
                            item = subset.sort_values(by='start', ascending=False).to_dict('records')[0]        #TODO: progressive subsetting if dimesion is not empty
                        else:
                            logger.warning(f'XBRL Tag not found for: {ticker}')
                            continue
                        rec = RecordMetadata(
                                cik = cik,
                                accn = accn,
                                form = filing.file_type,
                                account = acct_key,
                                value = item['value_concept'],                  
                                account_title = acct_rec.table_account,
                                xbrl_tag = xbrl_tag,
                                fy = fy,
                                fp = fp,
                                end = end,
                                filed = filing.file_date
                        )
                        recs.append(rec)
                    except Exception as e:
                        logger.error(e)
                        continue
                    
            # save to db
            df = pd.DataFrame(recs, columns=RecordMetadata._fields)
            try:
                df.to_sql(self.table_name[0]['name'], self.engine, index=False, if_exists='append')
            except sql.exc.IntegrityError:
                pass
            except Exception as e:
                logger.error(e)
        return True


    '''
    def populate_quarterly_filings(self):
        """Import quarterly (10-K/-Q) data from SEC EDGAR API."""
        pass


    def populate_earnings_releases(self):
        """Download, extract, and import earnings (8-K_ data 
        from SEC EDGAR API."""
        pass
    '''


    def get_earnings_releases(self, firms, after):
        """Get press releases (8-K/EX-99.*) and extract account values.
        """

        def check(row):
            """Mask for selecting press release data."""
            if type(row.Type) == str and ('EX-99.1' in row.Type or 'EX-99.2' in row.Type or 'EX-99.3' in row.Type):
                return True
            else:
                return False

        AFTER = after           #"2022-01-01"

        # check if table initialized
        stmt = "SELECT * FROM " + self.table_name[0]["name"] + " WHERE form like '8-%'"
        df_8k = pd.read_sql(stmt, self.engine)
        #if df_8k.shape[0] > 0: pass

        # download
        ciks = [str(firm._cik) for firm in firms]
        for firm in firms:
            TICKER = firm.get_info()['ticker']
            urls = self.downloader.get_metadata("8-K",
                                TICKER, 
                                after = AFTER)   

        df = self.downloader.filing_storage.get_dataframe(mode='document')
        sel1 = df[(df['short_cik'].isin(ciks)) 
                    & (df['file_type'] == '8-K')
                    & (pd.to_datetime(df['file_date']) > AFTER) 
                    ]
        mask = sel1.apply(check, axis=1)
        sel2 = sel1[mask]
        lst_of_idx = sel2.index.tolist()
        staged = self.downloader.filing_storage.get_document_in_record( lst_of_idx )
        updated_docs = self.downloader.get_documents_from_url_list(staged)

        accns = [doc[0].split('|')[1] for doc in updated_docs['new']]
        df_new_filings = df_8k[df_8k['accn'].isin(accns)]
        if set( df_new_filings['accn'].to_list() ) == set( accns ):
            return True
        else:
            # extract
            selected_docs = list(updated_docs['new']) + list(updated_docs['previous'])
            recs = []
            for key, doc in selected_docs:
                if not doc.FS_Location:
                    continue
                cik = doc.FS_Location.parent.parent.name             #TODO:<<<fix this craziness by creating a class and providing attributes: cik, accn
                target_firm = [firm for firm in firms if firm.get_info()['cik'].__str__()==cik][0]
                ticker = target_firm.get_info()['ticker']
                doc_meta = sel2[sel2.Document == doc.Document].to_dict('record')[0]

                # process with extractor
                items = self.extractor.execute_extract_process(doc=doc, ticker=ticker)
                items_key = list(items.keys())[0]
                for acct, val in items[items_key].items():
                    if type(val) == str: continue
                    if doc.Type != self.extractor.config[ticker].accounts[acct].exhibits: continue
                    scale = self.extractor.config[ticker].accounts[acct].scale
                    rec = RecordMetadata(
                                    cik = str(doc_meta['short_cik']),
                                    accn = str(doc_meta['accession_number']),
                                    form = f"{doc_meta['file_type']}/{doc.Type}",
                                    account = acct,
                                    value = scale_value(val, scale),
                                    account_title = self.extractor.config[ticker].accounts[acct].table_account,
                                    xbrl_tag = self.extractor.config[ticker].accounts[acct].xbrl,
                                    fy = np.nan,
                                    fp = np.nan,
                                    end = np.nan,
                                    filed = doc_meta['file_date']
                            )
                    recs.append(rec)

            # save to db
            df = pd.DataFrame(recs, columns=RecordMetadata._fields)
            try:
                df.to_sql(self.table_name[0]['name'], self.engine, index=False, if_exists='append')
            except sql.exc.IntegrityError:
                pass
            except Exception as e:
                logger.error(e)
        return True


    def format_raw_quarterly_records(self):
        """Prepare the raw quarerly record data for the `filing` table."""

        df_init = self.query_database(table_name='records')
        df = df_init[df_init['form'].isin(['10-K','10-Q'])]
        # prepare records and insert into db
        if df.shape[0] > 0:
            df_columns = df.drop(labels=["account","value","account_title","xbrl_tag"], axis=1).drop_duplicates(subset=["cik","accn"])
            df_wide = df.pivot(index=["cik","accn"], columns="account", values="value").reset_index()

            tmp_key = pd.DataFrame(  df.groupby(['cik','accn'])['account'].apply(list) )
            tmp_key['account_title'] = pd.DataFrame(  df.groupby(['cik','accn'])['account_title'].apply(list) )['account_title']
            tmp_key['xbrl_tag'] = pd.DataFrame(  df.groupby(['cik','accn'])['xbrl_tag'].apply(list) )['xbrl_tag']
            tmp_key['titles'] = tmp_key.apply(lambda row: str(list( zip(row['account'], row['account_title'], row['xbrl_tag']))), axis=1 )
            tmp_key['index'] = tmp_key.index
            tmp_key.reset_index(inplace=True)
            df_wide['titles'] = tmp_key['titles']

            df_wide_total = pd.merge(df_wide, df_columns, on=["cik","accn"], how="left")
            df_wide_total["filed"] = pd.to_datetime(df_wide_total["filed"])
            df_wide_total.sort_values(by="filed", ascending=False, inplace=True)
            df_wide_total['yr_qtr'] = df_wide_total.apply(create_qtr, axis=1)
            #TODO: apply FilingMetadata before importing
            try:
                df_wide_total.to_sql(self.table_name[1]['name'], 
                        con=self.engine,
                        if_exists="append", 
                        index=False
                        )
            except sql.exc.IntegrityError as e:
                logger.warning("Unique key violation on insert")
            else:
                logger.info(f"Inserted {df_wide_total.shape[0]} records to table {self.table_name[1]['name']}")
            return True
        else:
            return False


    def format_raw_earnings_records(self):
        """Format earnings records for ingest into `filings` table."""

        df_tmp = self.query_database(table_name='filings')
        df_10q = df_tmp[df_tmp['form'].isin(['10-K','10-Q'])]
        df_init = self.query_database(table_name='records')
        df = df_init[~df_init['form'].isin(['10-K','10-Q'])]
        # prepare table
        if df.shape[0] > 0:
            df_columns = df.drop(labels=['account','value'], axis=1).drop_duplicates(subset=['cik','accn','form'])
            #check for duplicates that will cause error: df[df.duplicated(subset=['cik','accn','acct'])==True].shape
            df_wide = df.pivot(index=['cik','accn','form'], columns='account', values='value').reset_index()

            tmp_key = pd.DataFrame(  df.groupby(['cik','accn'])['account'].apply(list) )
            tmp_key['account_title'] = pd.DataFrame(  df.groupby(['cik','accn'])['account_title'].apply(list) )['account_title']
            tmp_key['xbrl_tag'] = pd.DataFrame(  df.groupby(['cik','accn'])['xbrl_tag'].apply(list) )['xbrl_tag']
            tmp_key['titles'] = tmp_key.apply(lambda row: str(list( zip(row['account'], row['account_title'], row['xbrl_tag']))), axis=1 )
            tmp_key['index'] = tmp_key.index
            tmp_key.reset_index(inplace=True)
            df_wide['titles'] = tmp_key['titles']        

            df_wide_total = pd.merge(df_wide, df_columns, on=['cik','accn','form'], how='left')
            df_wide_total['filed'] = pd.to_datetime(df_wide_total['filed'])
            df_wide_total.sort_values(by='filed', ascending=False, inplace=True)
            df_8k = df_wide_total.copy(deep=True)

            if df_10q.shape[0] > 0:
                table_cols = df_10q.columns.to_list()
                #df_8k['yr_qtr'] = get_8k_qtr(df_8k, df_10q)                            #TODO: this may be too complicated
                report_qtr = pd.DatetimeIndex( df_8k['filed'] ) - timedelta(90)         #get previous quarter
                df_8k['yr_qtr'] = report_qtr.year.astype(str) + '-Q' + report_qtr.quarter.astype(str)
                df_8k['fy'] = df_8k['yr_qtr'].str.split(pat='-').str[0]
                df_8k['fp'] = df_8k['yr_qtr'].str.split(pat='-').str[1].replace({'1':'Q1','2':'Q2','3':'Q3','4':'FY'})
                current_cols = df_8k.columns
                select_cols = [col for col in current_cols if col in table_cols]
                df_to_commit = df_8k[select_cols]
            else:
                df_8k['fy'] = None
                df_8k['fp'] = None
                df_to_commit = df_8k

            # load into db
            try:
                df_to_commit.to_sql(self.table_name[1]['name'], 
                        con=self.engine,
                        if_exists='append', 
                        index=False
                        )
            except sql.exc.IntegrityError as e:
                logger.warning('Unique key violation on insert')
            else:
                logger.info(f'Inserted {df_8k.shape[0]} records to table {self.table_name[1]["name"]}')
            return True
        else:
            return False



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


    def query_database(self, table_name):
        '''
        meta.reflect(bind=engine)
        filings = meta.tables[table_name]
        s = filings.select()
        result = engine.execute(s)
        for row in result:
            logger.info(row)
        '''
        df = pd.read_sql_table(table_name, self.engine)
        return df