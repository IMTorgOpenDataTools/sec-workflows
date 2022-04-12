# SEC Workflows

Pull data from SEC EDGAR and maintain in a database.


## Quick Start

Ensure `IMTorg/sec-edgar-downloader` is available.

```
mkdir .lib
mv sec-edgar-downloader-feature-address_multiple_issues/ .lib/
pipenv install .lib/sec-edgar-downloader-feature-address_multiple_issues/.
pipenv install .
python /workspaces/Prj-sec_workflows/sec_workflows/main.py init
```

or 

`vscode > debugger > Python CLI`


## Development

To be used in any venv, port the depdendencies to requirements.txt: `pipenv run pip freeze > requirements.txt`



## TODO

* add extraction workflow
* init
  - check db before inserting new records
* run 
  - fix workflow
* ~~debug code in order~~
* ~~determine where `sec-edgar-downloader` will be added~~
  - ~~add api requests to module~~
* add logging
* format excel report
* integrate sec-edgar-downloader