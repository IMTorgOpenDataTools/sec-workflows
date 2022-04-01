#!/usr/bin/env python3
"""
Process to serve reports.

*** DEPRECATED ***

This is not maintained, but will be kept avaible for future use.


"""
__author__ = "Jason Beach"
__version__ = "0.1.0"
__license__ = "MIT"

#my libs

#third-party
import pandas as pd
#from sqlalchemy import create_engine
#from sqlalchemy import Table, Column, Integer, String, MetaData
#from sqlalchemy_utils import database_exists#, create_database
from flask import Flask, jsonify, request

#built-in
from pathlib import Path
import logging
import time
import requests
from collections import namedtuple



#constants
SERVER = "127.0.0.1"
PORT = 5000

#configure
logger = logging.getLogger(__name__)
app =   Flask(__name__)

app.config.update (
    debug = True,
    host = '0.0.0.0',
    port = PORT
    #SERVER_NAME = f'{SERVER}:{PORT}'
)


'''
def create_report(engine):
    """Create the report."""
    meta.reflect(bind=engine)
    filings = meta.tables[table_name]
    s = filings.select()
    result = engine.execute(s)
    for row in result:
        print (row)
    #df = pd.read_sql_table(table_name, engine)
'''

@app.route('/update_report', methods=['GET'])
def update_report():
    if(request.method == 'GET'):
        #create_report()
        data = {"data": True}
        return jsonify(data)

@app.route('/get_report', methods=['GET'])
def get_report():
    if(request.method == 'GET'):
        data = {"data": "This is the most-recent report"}
        return jsonify(data)




if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=PORT)