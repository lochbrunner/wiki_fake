# Wiki Fake

**Learning project**: This GAN aims to fake Wikipedia sentences.

## Setup

```zsh
python3 -m venv venv 
. venv/bin/activate
pip install -r requirements.txt
```

## Crawl data

```zsh
(cd wiki_crawler; scrapy crawl wiki -o data/sentences.pickle )
./preprocess.py -i data/sentences.pickle -o data/processed.pickle
```

You might have to interrupt the crawling (*Ctrl+c*) when you don't want to download all Wikipedia sites.

## Training

```zsh
./gan.py -i data/processed.pickle -m data/snapshot.pickle
```

Example

> genaue gruppenkommandanten studiert kontextdefinition abstrakt konsequenzoperation mehrsprachig


## Visualization

```zsh
./visu.py -i data/processed.pickle -m data/snapshot.pickle
```

## TODO

* Use [sacremoses](https://github.com/alvations/sacremoses) as tokenizer
  