#!/usr/bin/env python3
"""
Output class to manage all application output except reports.
"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#third-party
import email
import pandas as pd

#builtin
import shutil
from collections import namedtuple
import subprocess
from subprocess import PIPE, STDOUT
from enum import unique

from pathlib import Path
#from datetime import datetime, date, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#libs
import sys
sys.path.append(Path('config').absolute().as_posix() )
from _constants import (
    FILE_EMAILS
)




class Output:

    def __init__(self, emails_file_or_dictlist = None, email_network_drive = None, logger = None):
        if type(emails_file_or_dictlist) == list:
            self.emails_df = pd.DataFrame(emails_file_or_dictlist)
        elif type(emails_file_or_dictlist) == str:
            self.emails_df = pd.read_csv(emails_file_or_dictlist)
        else:
            logger.debug('Output class must be instantiated with file or dict-list')
            exit()
        self.email_network_drive = email_network_drive
        self.logger = logger


    def send_notification(self, error = False):
        """Send email notification that report is updated."""

        #constants
        subject = 'SEC 8-K Update'
        df = pd.read_csv(FILE_EMAILS)
        df_emails = df[df['notify'] == True]

        #scenarios
        template_success = (f'''
        Dear Sir / Ma'am, 
        This is a notification that the SEC Earnings report is updated.  You can find it in the following network drive: 
        {self.email_network_drive}
        ''')
        body_success = bytes(template_success, encoding='utf-8')
        emails_success = df_emails['address'].tolist()

        body_fail = b'''*** There was an error ***'''
        df_admin_only = df_emails[df_emails['admin'] == True]
        emails_fail = df_admin_only['address'].tolist()

        if not error:
            body_content = body_success
            emails = emails_success
        else:
            body_content = body_fail
            emails = emails_fail

        #execute
        checks = []
        bashCommand = ["mailx", "-s", subject, *emails]
        try:
            p = subprocess.Popen(bashCommand, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
            result = p.communicate(input=body_content)[0]
            checks.append(result)
        except:
            self.logger.error("failed to send email notification.")

        return checks