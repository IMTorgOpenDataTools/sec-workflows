#!/usr/bin/env python3
"""
Test database engine functionality.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
from sec_edgar_downloader import UrlComponent as uc

#builtin
import pytest
import sys
from pathlib import Path
import requests

#lib
from sec_workflows.database import Database
from sec_workflows.output import Output
from sec_workflows.utils import (
    delete_folder,
    api_request, 
    remove_list_dups,
    make_long_cik,
    scale_value,
    create_qtr,
    poll_sec_edgar
)
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    EMAIL_NETWORK_DRIVE,
    meta,
    LIST_ALL_TABLES,
    Logger,
    logger
)



output = Output(
    emails_file_or_dictlist=[{'name':'Joe Smith', 'address':'joe.smith@gmail.com', 'notify':True, 'admin':True}],
    email_network_drive = EMAIL_NETWORK_DRIVE,
    logger = logger
)






def test_send_notification_success():
    #this will fail if `mailx` is not available
    checks = output.send_notification(error=False)
    assert checks == []


def test_send_notification_failure():
    #this will fail if `mailx` is not available
    checks = output.send_notification(error=True)
    assert checks == []


def test_api_request():
    cik = 36104
    client = requests.Session()
    recent = api_request(session=client, type='firm_details', cik=cik, acct=None)
    assert len(recent['form']) == 1000


def test_remove_list_dups():
    lst = [
        {'cik':'1'},
        {'cik':'2'},
        {'cik':'3'},
        {'cik':'2'},
    ]
    new_lst = remove_list_dups(lst, 'cik')
    assert len(new_lst) == 3


def test_make_long_cik():
    cik = 36104
    long_cik = make_long_cik(cik)
    assert long_cik == '0000036104'


def test_scale_value():
    val = 1100
    scale = 'thousands'
    rtn = scale_value(val, scale)
    assert rtn == 1100000


def test_create_qtr():
    row = {
        'fy': '2022',
        'fp': 'Q1',
        'filed': '2022-01-01',
        'end': None
    }
    rtn = create_qtr(row)
    assert rtn == '2022-Q1'





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
    del db
    log_file.unlink() if log_file.is_file() else None
    db_file.unlink() if db_file.is_file() else None
    delete_folder(log_file.parent)


def test_poll_sec_edgar(mocker, resource_db):
    output_data = {
                'accessionNumber': ['0001193125-22-150841'],
                'filingDate': ['2022-05-16'],
                'form': ['8-K'],
                'size': [235273],
                'primaryDocument': ['d329362d8k.htm']
    }
    mock_request = mocker.patch('sec_workflows.utils.api_request', 
                return_value = output_data
                ) 
    firms = [ uc.Firm(ticker = "USB") ]
    after_date = '2022-05-01'
    changed_firms = poll_sec_edgar(resource_db, firms, after_date)
    mock_request.call_args.__str__() == "call(session=<requests.sessions.Session object at 0x7f065f9cbf70>, type='firm_details', cik=36104, acct=None)"
    assert changed_firms.__repr__() == "{'8k': {US BANCORP \DE\}, '10kq': set()}"