#!/usr/bin/env python3
"""
Test database engine functionality.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"


import pytest

import sys
from pathlib import Path
from datetime import date, datetime, timedelta
import requests

from sec_workflows.utils import Logger
from sec_workflows.database import Database

from sec_workflows.utils import (
    send_notification, 
    api_request, 
    remove_list_dups,
    make_long_cik,
    scale_value,
    create_qtr,
    poll_sec_edgar
)

sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    meta,
    LIST_ALL_FIRMS,
    LIST_ALL_TABLES
)



def test_send_notification():
    checks = send_notification()
    assert checks != []


def test_api_request():
    cik = 36104
    client = requests.Session()
    recent = api_request(session=client, type='firm_details', cik=cik, acct=None)
    assert len(recent['form']) == 1001


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
                    logger = logger
                    )
    # tests
    yield db

    # tear-down
    del db
    log_file.unlink() if log_file.is_file() else None
    db_file.unlink() if db_file.is_file() else None
    log_file.parent.rmdir()


def test_poll_sec_edgar(resource_db):
    days = timedelta(days = 3)
    start_date = datetime.now().date() - days
    after_date = start_date.strftime("%Y-%m-%d")

    changed_firms = poll_sec_edgar(resource_db, LIST_ALL_FIRMS, after_date)
    assert True == True