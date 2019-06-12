import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


global _model

def get_top_n_tfidf(response, feature_names, n=5):
    d = {}
    for col in response.nonzero()[1]:
        d[feature_names[col]] = float(response[0, col])
    return sorted(d, key=lambda key: d[key], reverse=True)[:n]


def train(dataset,arguments,lang='dutch'):
    nltk.download('stopwords')
    global _model
    stopwords = nltk.corpus.stopwords.words(lang)
    _model = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 3))
    _model.fit_transform(dataset)



def test(text,arguments, n=5):
    response = _model.transform([text])
    feature_names = _model.get_feature_names()
    return get_top_n_tfidf(response, feature_names, n)
