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


* run
  - request update page
  - determine if any target firms were updated
  - use firms with new db call (select call for 8k or 10k) with latest point in time
  - db is updated
  - report is updated
  - email notifications
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

