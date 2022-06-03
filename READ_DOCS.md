
# Background Documentation

This application combines multiple projects to obtain financial and economic data, then transforms and organizes it for easy use in a variety of workflows.


## Data Sources

* SEC EDGAR
  - firm information: name, cik, ticker, etc.
  - daily filings
  - firm filing details: data and exhibits
  - batch archive
* Earnings Call transcripts
  - most recent
  - batch archive
* Daily stock / bond yield
  - current
  - batch archive
* Economic indicators
  - current calendar
  - economic
  - firm


## Class structure

General overcview

```mermaid
classDiagram
    class Firm
    Firm : +cik
    Firm : +ticker
    Firm : +List~Report~ filings
    Firm : +List~Program~ portfolio
    Firm : +Person management
    Firm : +List~Security~ portfolio
    Firm : +valuation()
    Firm : +risk()
    Report <|-- SecFiling
    Report <|-- EarningsCall
    Program <|-- Loan
    Program <|-- Project
    Security <|-- FixedIncome
    Security <|-- Equity
    class Person
```

Current work

```mermaid
classDiagram
class Report
    Report: +metadata
    Report: +structured_XBRL
    Report: +structured_EXCEL
    Report: +structured_Table
    Report: +unstructured
    Report: +create_intermediate_file()
    Report: +create_final_file()
```





## References

* [sec-edgar-downloader](https://github.com/jadchaar/sec-edgar-downloader)
* ixbrl, xbrl, html
  - [ixbrlparse]()
  - [python-xbrl]()
  - [py-xbrl]()
* [mermaid diagrams](https://mermaid-js.github.io)