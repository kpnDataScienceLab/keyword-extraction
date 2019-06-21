import nltk
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

global _freq_matrix  # for each word, stores the N of documents that contain it
global _vectorizer
global _l_avg
global _n_docs


def score_bm25(tf_dt, l_d, l_avg, n_docs, dft):
    """
    BM25 with the second term removed. That term regulates the importance of words
    given how often they occur in a query, but since we're dealing with single word queries
    it's unnecessary.
    :param tf_dt: Frequency of term T in document D
    :param l_d: Length of document D
    :param l_avg: Average length for all documents
    :param n_docs: Total number of documents
    :param dft: Number of documents that contain term T
    :return: The BM25 score for term T in document D
    """
    # tunable parameters
    k1 = 1.5
    b = 0.75

    # equation
    k = k1 * ((1 - b) + b * (l_d / l_avg))
    normalizing_term = ((k1 + 1) * tf_dt) / (k + tf_dt)
    idf = log(n_docs / dft)
    return normalizing_term * idf


def train(dataset, arguments, lang='dutch'):
    """
    Process the given text along with the full dataset to produce a word vocabulary
    and a frequency matrix.
    :param dataset: List of texts that the model will be trained on
    :param lang: Language to be used for the stopwords
    """
    global _freq_matrix
    global _vectorizer
    global _l_avg
    global _n_docs

    # get list of dutch stopwords
    stopwords = nltk.corpus.stopwords.words(lang)

    # get word list and frequency matrix (an ndarray where rows=documents and cols=words)
    _vectorizer = CountVectorizer(stop_words=stopwords,
                                  strip_accents='unicode',
                                  ngram_range=(1, 3))
    freq_matrix = _vectorizer.fit_transform(dataset)

    # global parameters
    _l_avg = np.mean(freq_matrix.sum(axis=1))
    _n_docs = freq_matrix.shape[0]

    n_words = freq_matrix.shape[1]

    _freq_matrix = {}
    for i, word in tqdm(enumerate(_vectorizer.get_feature_names()), total=n_words, ncols=80):
        _freq_matrix[word] = freq_matrix.getcol(i).count_nonzero()


def test(text, arguments=None, k=40, lang='dutch'):
    """
    :param text: Text that the keywords are extracted from
    :param n: Number of keywords to return. Use -1 to return all of them
    :param lang: Language for the stopwords
    :return: Top n keywords, ordered from most to least relevant
    """
    global _freq_matrix
    global _vectorizer
    global _l_avg
    global _n_docs

    # check model has been trained
    if _freq_matrix is None:
        print("[WARNING] BM25 hasn't been trained! Returning nothing...")
        return []

    # get frequency matrix for the text
    text_freq_matrix = _vectorizer.fit_transform([text])
    text_freq_matrix = text_freq_matrix.toarray()

    # length of the current document
    l_d = sum(text_freq_matrix[0])

    # make text frequency matrix into a dictionary
    freq_matrix = {}
    for i, word in enumerate(_vectorizer.get_feature_names()):
        freq_matrix[word] = text_freq_matrix[0, i]

    n_docs = _n_docs + 1

    # compute bm25 scores
    scores = []
    for word in freq_matrix:
        # term level
        tf_dt = freq_matrix[word]
        dft = _freq_matrix.get(word, 0)
        s = score_bm25(tf_dt, l_d, _l_avg, n_docs, dft)
        scores.append((word, s))

    # reorder and extract keywords
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [pair[0] for pair in scores]

    return keywords[0:k] if len(keywords) >= k else keywords
