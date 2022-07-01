#!/usr/bin/env python3
"""
Process to initialize and listen for updates to the SEC EDGAR Database.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party

#built-in
from pathlib import Path
import time
from datetime import date, datetime, timedelta
import argparse

#my libs
from database import Database
from report import Report
from output import Output
from utils import (
    poll_sec_edgar,
    reset_files
)
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    FILE_EMAILS,
    FILE_LOG,
    FILE_DB,
    EMAIL_NETWORK_DRIVE,
    meta,
    LIST_ALL_FIRMS,
    LIST_ALL_TABLES,
    MINUTES_BETWEEN_CHECKS,
    DIR_REPORTS,
    DIR_SEC_DOWNLOADS,
    logger
)


#configure
db = Database(db_file = FILE_DB,
                tables_list = LIST_ALL_TABLES,
                meta = meta,
                logger = logger,
                path_download = DIR_SEC_DOWNLOADS
                )
report = Report(db = db, 
                output_path=DIR_REPORTS
                )
output = Output(
                emails_file_or_dictlist=FILE_EMAILS,
                email_network_drive = EMAIL_NETWORK_DRIVE,
                logger = logger
)


#processes
def initialize_process():
    years = timedelta(weeks = 52)
    start_date = datetime.now().date() - years
    after_date = start_date.strftime("%Y-%m-%d") 

    api = db.get_quarterly_statements(LIST_ALL_FIRMS, after_date)
    scrape = db.get_earnings_releases(LIST_ALL_FIRMS, after_date)
    format_10q = db.format_raw_quarterly_records()
    format_8k = db.format_raw_earnings_records()

    if all([api, scrape, format_10q, format_8k]): 
        report.create_report(type='long')
        logger.info(f'Database initialization complete')
    else:
        logger.info(f'No databae update necessary')


def run_process():
    loop = True                                                                 #setup based on desired loop style
    while loop:
        days = timedelta(days = 3)
        start_date = datetime.now().date() - days
        after_date = start_date.strftime("%Y-%m-%d")
        start_records = db.query_database('records').shape[0]

        changed_firms = poll_sec_edgar(db, LIST_ALL_FIRMS, after_date)
        firms = []
        values = list( changed_firms.values() )
        [firms.extend(list(item)) for item in values]
        if len(firms) > 0:
            logger.info('sec edgar changed')
            if changed_firms['10kq']:
                firm_list = list(changed_firms['10kq'])
                api = db.get_quarterly_statements(firm_list, after_date)
                format_10q = db.format_raw_quarterly_records()
                logger.info('database updated 10kq')
            if changed_firms['8k']:
                firm_list = list(changed_firms['8k'])
                scrape = db.get_earnings_releases(firm_list, after_date)
                format_8k = db.format_raw_earnings_records()
                logger.info('database updated 8k')

            end_records = db.query_database('records').shape[0]
            if end_records > start_records:
                report.create_report(type='accounting_policy')
                output.send_notification(error=False)
        else:
            logger.info('no change to server')
        secs = MINUTES_BETWEEN_CHECKS * 60
        time.sleep(secs)
        loop = False


def reports_process():
    report.create_report(type='long')
    report.create_report(type='accounting_policy')
    #create_report(type='trend', db=db, output_path=DIR_REPORTS)
    report.create_report(type='validate')






def main(args):
    """Application entrypoint"""
    logger.info(f'Starting process in {args.mode[0]} mode')

    #check db file and schema
    check_db = db.check_db_file()
    check_schema = db.check_db_schema()
    checks = [check_db, check_schema]
    if not all(checks):
        logger.warning("DB checks failed.")
        exit()

    #process
    match args.mode[0]:
        case 'init':
            initialize_process()
        case 'run':
            run_process()
        case 'report':
            reports_process()
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