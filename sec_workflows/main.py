#!/usr/bin/env python3
"""
Process to initialize and listen for updates to the SEC EDGAR Database.
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
import os
from pathlib import Path
import logging
import time
from datetime import date
import requests
import argparse

#my libs
from database import Database
from utils import (
    poll_sec_edgar,
    initialize_db,
    create_report,
    get_press_releases,
    reset_files
)
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    firms,
    log_file,
    db_file,
    tables_list,
    MINUTES_BETWEEN_CHECKS,
    QUARTERS_IN_TABLE,
    OUTPUT_REPORT_PATH,
    meta,
    filings,
    FilingMetadata
)


#configure
#Path(log_file).mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)







def main(args):
    """Application entrypoint
    """
    logger.info(f'Starting process in {args.mode[0]} mode')

    #configure
    db = Database(db_file = db_file,
                    tables_list = tables_list,
                    meta = meta,
                    logger = logger
                    )

    #check db file
    check_db = db.check_db_file()

    #check db schema
    check_schema = db.check_db_schema()

    #process
    match args.mode[0]:
        case 'init':
            api = initialize_db(db, firms)
            scrape = get_press_releases(db, firms)
            # TODO: add those to `raw_` files, then combine, here
            if api and scrape: 
                create_report(report_type='long', db=db, output_path=OUTPUT_REPORT_PATH)
                logger.info(f'Database initialization complete')
            else:
                logger.info(f'No databae update necessary')
        case 'run':
            while True:
                changed_data = poll_sec_edgar(db, ciks)     #TODO: update with with firms
                if changed_data:
                    print("sec edgar changed")
                    db.update_database()
                    print("database updated")
                    create_report(report_type='long', db=db, output_path=OUTPUT_REPORT_PATH)
                    print("report created")
                else:
                    print("no change to server")
                secs = MINUTES_BETWEEN_CHECKS * 60
                time.sleep(secs)
        case 'report':
            create_report(report_type='long', db=db, output_path=OUTPUT_REPORT_PATH)
            create_report(report_type='accounting_policy', db=db, output_path=OUTPUT_REPORT_PATH)
            #create_report(report_type='trend', db=db, output_path=OUTPUT_REPORT_PATH)
            #create_report(report_type='validate', db=db, output_path=OUTPUT_REPORT_PATH)
        case 'RESET_FILES':
            reset_files()

    logger.info(f'Process exited')
                    



if __name__ == "__main__":
    """Main entry point to the application.

    App start is executed in one of the following two modes:
      * `init`ialize system by creating new db with records or validate current records if db is present
      * `run` the listener for updates (8-K, 10-K/-Q), periodically, then create a report and output to directory
    
    ....
    """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("mode",
                        default='init',
                        nargs=1,
                        choices=['init', 'run', 'report', 'RESET_FILES'], 
                        help="`init`ialize or `run` the process"
                        )
    args = parser.parse_args()

    main(args)