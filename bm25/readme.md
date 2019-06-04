# BM25

### Requirements

To install the required packages use:
```
pip install -r requirements.txt
```
If needed, install the nltk stopwords dataset:
```
python -c "import nltk; nltk.download('stopwords')"
```

### Usage

To run the BM25 keyword extractor, navigate to the root
folder `keyword-extraction/`. Example usage:

```python
from bm25.bm25 import bm25

data = pd.read_csv('aligned_epg_transcriptions_npo1_npo2.csv')
text = data['text'][0]

# n is the amount of keywords to return
# the keywords are ordered from most to least relevant
bm25_words = bm25(text, n=5)
```