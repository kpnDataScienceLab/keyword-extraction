import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
file_name = 'aligned_epg_transcriptions_npo1_npo2.csv'
data = pd.read_csv(file_name)
texts = data['text']

stopwords = nltk.corpus.stopwords.words('dutch')
tfidfVec = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 3))
tfs = tfidfVec.fit_transform(texts)


def get_top_n_tfidf(response, feature_names, n=5):
    d = {}
    for col in response.nonzero()[1]:
        d[feature_names[col]] = float(response[0, col])
    return sorted(d, key=lambda key: d[key], reverse=True)[:n]


def tfidf(text, n=5):
    response = tfidfVec.transform([text])
    feature_names = tfidfVec.get_feature_names()
    return get_top_n_tfidf(response, feature_names, n)
