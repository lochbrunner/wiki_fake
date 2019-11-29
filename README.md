# Wiki Fake

Fake wikipedia sentences.

## Setup

```zsh
python3 -m venv venv 
. venv/bin/activate
pip install -r requirements.txt
```

## Crawl data

```zsh
scrapy crawl wiki -o data/sentences.csv
```
