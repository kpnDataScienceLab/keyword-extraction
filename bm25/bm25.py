import pandas as pd
import nltk
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer


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


def fit(text, stopwords, n_docs=0):
    # load texts
    file_name = 'aligned_epg_transcriptions_npo1_npo2.csv'
    data = pd.read_csv(file_name)

    texts = list(data['text'])
    texts.insert(0, text)

    # reduce amount of documents
    # if n_docs:
    #     assert n_docs > 0
    #     texts = texts[:n_docs]

    # get word list and frequency matrix (an ndarray where rows=documents and cols=words)
    vectorizer = CountVectorizer(stop_words=stopwords,
                                 strip_accents='unicode',
                                 ngram_range=(1, 3))
    freq_matrix = vectorizer.fit_transform(texts)

    return vectorizer.get_feature_names(), freq_matrix.toarray()


def bm25(text, n=5):
    # get list of dutch stopwords
    stopwords = nltk.corpus.stopwords.words('dutch')

    # get statistics for the data. the text is at position 0 in the frequency matrix
    words, freq_matrix = fit(text, stopwords)

    # global parameters
    l_avg = np.mean(freq_matrix.sum(axis=1))
    n = freq_matrix.shape[0]

    # document parameters
    l_d = sum(freq_matrix[0])

    # indices of the possible document keywords
    keyword_idxs = np.nonzero(freq_matrix[0])[0].tolist()

    # compute bm25 scores
    scores = []
    for idx in keyword_idxs:
        # term level
        tf_dt = freq_matrix[0, idx]
        dft = np.count_nonzero(freq_matrix[:, idx])
        s = score_bm25(tf_dt, l_d, l_avg, n, dft)
        scores.append((words[idx], s))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [pair[0] for pair in scores]

    return keywords[0:n] if len(keywords) >= n else keywords
