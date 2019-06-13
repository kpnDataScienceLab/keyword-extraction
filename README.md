# UvA Keyword Extraction Project

### Requirements

To install the required packages use:
```
$ pip install -r requirements.txt
```
If needed, install the nltk stopwords dataset:
```
$ python -c "import nltk; nltk.download('stopwords')"
```

## Usage

To run the pipeline in its current state, run:

```
$ python pipeline.py [--tfidf] [--bm25] [--yake] [--rake] [--k] [--dataset]
```

The `--k` flag indicates the maximum number of keywords retrieved per
text, and the `--dataset` flag indicates which dataset to use out of,
at present, *500N-KPCrowd*, *DUC-2001*, and *Inspec*.

The other flags don't require arguments and specify whether to use that
method during the current run. Multiple methods may be used in the same
run.