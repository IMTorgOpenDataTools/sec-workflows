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




class Report:
    """Report class."""

    def __init__(self, type):
        self.type = type