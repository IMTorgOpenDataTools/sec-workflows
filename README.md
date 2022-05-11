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

* run with all banks and get new accounting policy report
* create config with topic (ACL) and list of associated xbrl tags
* init
  - save to `records` so less work if redo
  - check db `records` before inserting new records


