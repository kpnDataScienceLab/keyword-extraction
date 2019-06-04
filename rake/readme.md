# Rake

### Requirements

Install nltk:
```
pip install rake-nltk
```
If needed, install the nltk stopwords dataset:
```
python -c "import nltk; nltk.download('stopwords')"
```

### Usage

To run the rake keyword extractor, load a text and run it through the `rake()` function:

```python
from rake import rake

data = pd.read_csv('../aligned_epg_transcriptions_npo1_npo2.csv')
text = data['text'][0]


```

### Sources

The medium stopword list was found on https://countwordsfree.com/stopwords/dutch.
The large stopword list was found on https://eikhart.com/blog/dutch-stopwords-list