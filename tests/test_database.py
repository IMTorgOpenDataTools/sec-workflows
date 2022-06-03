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
from sec_workflows.utils import Logger
from sec_workflows.database import Database

sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    LIST_ALL_TABLES,
    meta
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
                    logger = logger
                    )
    # tests
    yield db

    # tear-down
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