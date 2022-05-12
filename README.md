# SEC Workflows

Pull data from SEC EDGAR and maintain in a database.


## Quick Start

Add the config file: `config/Firm_Account_Info.csv`.  Ensusre `config/ciks.csv` is appropriate.

Ensure `IMTorg/sec-edgar-downloader` is available.

```
mkdir .lib
mv sec-edgar-downloader-feature-address_multiple_issues/ .lib/
pipenv install .lib/sec-edgar-downloader-feature-address_multiple_issues/.
pipenv install .
python /workspaces/Prj-sec_workflows/sec_workflows/main.py init

pipenv install .lib/sec-edgar-extractor-dev/.
pipenv install -r .lib/sec-edgar-extractor-dev/requirements.txt 
```


or 

`vscode > debugger > Python CLI`


## Development

To be used in any venv, port the depdendencies to requirements.txt: `pipenv run pip freeze > requirements.txt`



## TODO

* automate: create config with topic (ACL) and list of associated xbrl tags (maybe across history of bank filings)
  - similarity ranking across xbrl tags
* init
  - why are 8k, 10k consistent across firms?
  - save intermediate to `records` so less work if redo
  - check db `records` before inserting new records
  - fix cfg
* report: 
  - wksheet-1 why are 8k, 10k consistent across firms?
  - wksheet-2 definitions (website, gaap taxonomy .xml)
* run
  - just do it
  - email notifications
* extractor
  - none type exceptions
  - 10+sec execution: bac
  - bac Loans
  - confidence


