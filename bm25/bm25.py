import pandas as pd
import nltk
import string
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer


def get_freq_matrix(texts):
    # get list of dutch stopwords
    stopwords = nltk.corpus.stopwords.words('dutch')

    # get frequency matrix
    # TODO: try using ngram_range=(1, 3)
    vectorizer = CountVectorizer(stop_words=stopwords)
    freq_matrix = vectorizer.fit_transform(texts)

    return vectorizer.get_feature_names(), freq_matrix.toarray()


def score_bm25(tf_dt, l_d, l_avg, n, dft):
    """
    BM25 with the second term removed. That term regulates the importance of words
    given how often they occur in a query, but since we're dealing with single word queries
    it's unnecessary.
    """
    # tunable parameters
    k1 = 1.5
    b = 0.75

    # equation
    k = k1 * ((1 - b) + b * (l_d / l_avg))
    normalizing_term = ((k1 + 1) * tf_dt) / (k + tf_dt)
    idf = log(n / dft)
    return normalizing_term * idf


def fit(n_docs=30):
    # load texts
    file_name = '../aligned_epg_transcriptions_npo1_npo2.csv'
    data = pd.read_csv(file_name)

    texts = data['text']
    if n_docs:
        assert n_docs > 0
        texts = texts[:n_docs]

    # get word list and frequency matrix (an ndarray where rows=documents and cols=words)
    words, freq_matrix = get_freq_matrix(texts)
    return words, freq_matrix


def preprocess(text):
    t = text.translate(None, string.punctuation).lower()
    return t


def bm25(text, n=5):
    words, freq_matrix = fit()
    pass