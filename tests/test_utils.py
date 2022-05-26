import pytest

import os
import shutil
from bs4 import BeautifulSoup
import requests

from sec_workflows.utils import (
    send_notification, 
    api_request, 
    remove_list_dups,
    make_long_cik,
    scale_value,
    create_qtr
)



def test_send_notification():
    check = send_notification()
    assert check == True


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