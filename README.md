# SEC Workflows

Pull data from SEC EDGAR and maintain in a database for report creation.


## Quick Start

Typical configuration tasks in `config/`: 

* `ciks.csv` contain correct bank cik for global var `FILE_FIRMS`
* `emails.csv` used for appropriate notifications
* `_constants.py, DIR_REPORTS` for reports to specific directory
* a new `logs/` directory can be added
* addditional `_constants.py` paths that may need modifying:
  - `DIR_SEC_DOWNLOADS`
  - `FILE_LOG`
  - `FILE_DB`

The basic workflow includes:

* Initialize the database: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py init`
* Run check for SEC Filing updates and populate database with them: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py run`
* To create all available reports: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py reports`



## Setup and Install 

Ensure modules `IMTorg/sec-edgar-downloader` and `IMTorg/sec-edgar-extractor` are available.  Install using the following.

```
mkdir .lib
mv sec-edgar-downloader-feature-address_multiple_issues/ .lib/
pipenv install -e .lib/sec-edgar-downloader/.
pipenv install -e .lib/sec-edgar-extractor-dev/.
```

Install typical dependencies: `pipenv install -r .lib/sec-edgar-extractor-dev/requirements.txt `

Install [`tidy` for linux](https://www.html-tidy.org/), source code is [here](https://github.com/htacg/tidy-html5).



## Development

The following command can be used in any packaging tool or venv, 

* port the depdendencies to requirements.txt: `pipenv run pip freeze > requirements.txt` or transform the Pipfile
* remove and reinstall the two `sec-edgar-*` module entries
* add to venv with: `pip install -r requirements.txt`

The extraction process (when run operationally) creates intermediate files for all account topics.  This provides improved processing if database population is performed, again.  However, if the `Firm_Account_Info.csv` extraction configuration file is changed, then all the extractions will have to be removed to enable the process to be re-run with the new configuration.  

This is performed with the following: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py RESET_FILES`

This will only remove the intermediate files, so the developer may also want to remove the `archive/prod.db`, `archive/process.log` files, as well.



## Testing

Use the following to ensure tests are configured, then run the tests.

```
pipenv run pip list
pytest --fixtures
pipenv install pytest-mock           #must be available
pytest --collect-only
pytest
```


## Description of Steps

TODO



## TODO

* phase-II:automated account discovery
  - ~~update config/Firm_Account_Info.csv to reflect current use~~
  - mod config/Firm_Account_Info.csv: topic-based, incorporate timespans for each topic-title-tag (reporting changes over time)
  - use tag's name, definition, codification to determine similarity amongst all xbrl tags
  - determine best fit xbrl tag for specific topic, for any firm
  - get report filing title for given xbrl tag
* deployment / lsf, grid: tidy not being found
* workflow
  - database.py decompose parts, determine through tests
  - ~~how does report timespan increase? does it append new, or all-or-nothing?~~ => append new
* deployment
  - ~~checklist~~
* confidence level: deviation from trend
* report: validation
  - 10q should be validation '1', note difference between validation of 8k
  - cik 19617, 40729 has only 1 time period record
* downloader and extractor
  - add logger to downloader, extractor
  - replace Filing, Firm, ... automated populating from web via instantiation with a `.initialize()` method to do it explicitly
* extractor
  - Exception: 'NoneType' object is not iterable
  - 10+sec execution: bac
  - bac Loans
  - confidence based on steps in process
  - wksheet-2 definitions (website, gaap taxonomy .xml)
* report: schema and automated creation


## References

* [cmdln python debugger, pdb](https://qxf2.com/blog/debugging-in-python-using-pytest-set_trace/)
* [sec edgar taxonomy info](https://sec.gov/info/edgar/edgartaxonomies.shtml)