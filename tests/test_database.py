#!/usr/bin/env python3
"""
Test database engine functionality.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
import numpy as np
from sec_edgar_downloader import UrlComponent as uc
#from sec_edgar_downloader import Downloader
#from sec_edgar_downloader import FilingStorage
from sec_edgar_downloader import _constants

#builtin
import pytest
import sys
from pathlib import Path
import datetime

#lib
from sec_workflows.database import Database
from sec_workflows.utils import (
    delete_folder,
    poll_sec_edgar
)
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    LIST_ALL_TABLES,
    meta,
    RecordMetadata,
    Logger
)






@pytest.fixture()
def resource_db():
    # setup
    log_file = Path('./tests/tmp/process.log')
    db_file = Path('./tests/tmp/test.db')
    logger = Logger(log_file).create_logger()
    db = Database(db_file = db_file,
                    tables_list = LIST_ALL_TABLES,
                    meta = meta,
                    logger = logger,
                    path_download = './tests/tmp/downloads'
                    )
    # tests
    yield db

    # tear-down
    delete_folder( db.path_download )
    del db
    log_file.unlink() if log_file.is_file() else None
    db_file.unlink() if db_file.is_file() else None
    log_file.parent.rmdir()
  



class TestResourceDb:

    def test_check_db_file(self, resource_db):
        # check after file creation
        check_db = resource_db.check_db_file()
        assert check_db == True

    def test_validate_db_records(self, resource_db):
        check_records_exist = resource_db.validate_db_records()
        assert check_records_exist == False

    def test_check_db_schema(self, resource_db):
        check_schema = resource_db.check_db_schema()
        assert check_schema == True

    def test_get_quarterly_statements(self, resource_db):
        firms = [ uc.Firm(ticker = "USB") ]
        after_date = '2022-05-01'

        check_schema = resource_db.check_db_schema()
        execute = resource_db.get_quarterly_statements(firms, after_date)
        assert execute == True

    def test_get_earnings_releases(self, resource_db):
        firms = [ uc.Firm(ticker = "USB") ]
        after_date = '2022-05-01'

        check_schema = resource_db.check_db_schema()
        execute = resource_db.get_earnings_releases(firms, after_date)
        assert execute == True

    def test_change_in_10k_filing(self, resource_db, mocker):
        """Many resources and mocks."""
        #mock: poll_sec_edgar()
        output_data = {
                    'accessionNumber': ["0001193125-22-138788"],
                    'filingDate': ["2022-05-03"],
                    'form': ['10-Q'],
                    'size': [21073821],
                    'primaryDocument': ["d306477d10q.htm"]
        }
        mock_request = mocker.patch('sec_workflows.utils.api_request', 
                                    return_value = output_data
                                    ) 

        #mock: self.downloader.get_metadata(filing, TICKER, AFTER)
        output_doc = ('36104|0001193125-22-048709|186.0',
                        _constants.DocumentMetadata(
                            Seq = 186.0,
                            Description = 'EXTRACTED  XBRL INSTANCE DOCUMENT',
                            Document = 'd256232d10k_htm.xml',
                            Type = 'XML',
                            Size = 8095106,
                            URL = '/Archives/edgar/data/36104/000119312522048709/d256232d10k_htm.xml',
                            Extension = 'xml',
                            FS_Location = Path('tests/tmp/downloads/sec-edgar-filings/36104/0001193125-22-048709/d256232d10k_htm.xml')
                            )
                    )
        lst = [output_doc]
        output_dict = {'new': lst, 'previous':[], 'fail':[]}
        mock_downloader = mocker.patch('sec_workflows.database.Database.select_filing_records_for_download',
                                        return_value = output_dict
                                        )
        
        #mock: filing_storage
        output_recs = []
        rec = RecordMetadata(
                cik = '36104',
                accn = '0001193125-22-048709',
                form = '10-K',
                account = 'ACL',
                value = 6155000000,                
                account_title = 'Total allowance for credit losses',
                xbrl_tag = 'FinancingReceivableAllowanceForCreditLosses',
                fy = '2021',
                fp = 'FY',
                end = '2021-12-31',
                filed = datetime.datetime(2022, 2, 22, 0, 0)
        )
        output_recs.append(rec)
        mock_file_storage = mocker.patch('sec_workflows.database.Database.extract_values_and_load_into_record',
                                        return_value = output_recs
                                        )
        

        tgt_firms= [ uc.Firm(ticker = "USB") ]
        after_date = '2022-01-01'
        check_schema = resource_db.check_db_schema()

        changed_firms = poll_sec_edgar(resource_db, tgt_firms, after_date)
        firms = []
        values = list( changed_firms.values() )
        [firms.extend(list(item)) for item in values]
        if len(firms) > 0:
            if changed_firms['10kq']:
                firm_list = list(changed_firms['10kq'])
                api = resource_db.get_quarterly_statements(firm_list, after_date)

        from_test = list(rec._asdict().values())
        recs = resource_db.engine.execute("SELECT * FROM records WHERE form like '10-%'").fetchall()
        from_sql = list(recs[0])
        assert from_test[:-1] == from_sql[:-1]

    def test_change_in_8k_filing(self, resource_db):
        #mock poll_sec_edgar()
        #mock self.downloader.get_metadata(filing, TICKER, AFTER)
        #test db.get_quarterly_statements()
        assert True == True