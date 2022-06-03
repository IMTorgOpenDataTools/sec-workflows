# SEC Workflows

Pull data from SEC EDGAR and maintain in a database for report creation.


## Quick Start

Typical configuration tasks in `config/`: 

* `ciks.csv` contain correct bank cik 
* `emails.csv` used for appropriate notifications
* `_constants.py, report_output_path` for reports to specific directory
* addditional `_constants.py` paths include:
  - `DIR_SEC_DOWNLOADS`
  - `FILE_LOG`
  - `FILE_DB`

Initialize the database: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py init`

Run check for SEC Filing updates and populate database with them: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py run`

To make all available reports: `python /workspaces/Prj-sec_workflows/sec_workflows/main.py reports`



## Setup and Install 

Ensure modules `IMTorg/sec-edgar-downloader` and `IMTorg/sec-edgar-extractor` are available.  Install using the following.  Be certain to use the `-e` editable option; otherwise, the `config/Firm_Account_Info.csv` will not be available:

```
mkdir .lib
mv sec-edgar-downloader-feature-address_multiple_issues/ .lib/
pipenv install -e .lib/sec-edgar-downloader/.
pipenv install -e .lib/sec-edgar-extractor-dev/.
```

Install typical dependencies: `pipenv install -r .lib/sec-edgar-extractor-dev/requirements.txt `

Install [`tidy` for linux](https://www.html-tidy.org/), source code is [here](https://github.com/htacg/tidy-html5).

Prepare the following variables:

* Firm_Account_Info.csv
* ciks.csv
* `touch archive/process.log`
* db file
* emails file


or 

`vscode > debugger > Python CLI`




## Development

To be used in any venv, port the depdendencies to requirements.txt: `pipenv run pip freeze > requirements.txt`



## TODO

* automate: create config with topic (ACL) and list of associated xbrl tags (maybe across history of bank filings)
  - similarity ranking across xbrl tags
* extractor
  - Exception: 'NoneType' object is not iterable
  - 10+sec execution: bac
  - bac Loans
  - confidence
  - wksheet-2 definitions (website, gaap taxonomy .xml)
* report: 
  - wksheet-2 definitions (website, gaap taxonomy .xml)