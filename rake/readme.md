# Rake

### Requirements

To install the required packages use:
```
$ pip install -r requirements.txt
```
If needed, install the nltk stopwords dataset:
```
$ python -c "import nltk; nltk.download('stopwords')"
```

### Usage

To run the rake keyword extractor, navigate to the root
folder `keyword-extraction/`. Example usage:

```python
from rake.rake import rake

data = pd.read_csv('aligned_epg_transcriptions_npo1_npo2.csv')
text = data['text'][0]

# n is the amount of keywords to return
# the keywords are ordered from most to least relevant
rake_words = rake(text, n=5)
```

### Sources

The medium stopword list was found on https://countwordsfree.com/stopwords/dutch.
The large stopword list was found on https://eikhart.com/blog/dutch-stopwords-list