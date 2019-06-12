import pandas as pd
import nltk
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer


global words
global freq_matrix


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


def fit(texts, text, stopwords):
    """
    Process the given text along with the full dataset to produce a word vocabulary
    and a frequency matrix.
    :param texts: List of texts that the model will be trained on
    :param text: Text that will be used for keyword extraction
    :param stopwords: List of stopwords to be removed from all texts
    :return: The word vocabulary from the text collection, and a frequency matrix for those words
        (i.e. a matrix where rows=documents and cols=words)
    """
    # insert the document at the top of the list
    texts.insert(0, text)

    # get word list and frequency matrix (an ndarray where rows=documents and cols=words)
    vectorizer = CountVectorizer(stop_words=stopwords,
                                 strip_accents='unicode',
                                 ngram_range=(1, 3))
    freq_matrix = vectorizer.fit_transform(texts)

    return vectorizer.get_feature_names(), freq_matrix.toarray()


def remove_redundancy(keywords):
    """
    For each keyword in the keywords list, remove redundant keywords with a lower score compared
    to the original. This is done by comparing keywords which length differs by only one word,
    and checking for overlap.
    :param keywords: Ordered list of keywords
    :return: Filtered list of keywords
    """
    k_lens = [len(k.split(' ')) for k in keywords]

    c_idx = 0
    while c_idx < len(keywords):
        idx = 0
        while idx < len(keywords):
            # check that the length of the two keywords differs by only one,
            # and check whether one is a substring of the other
            if (k_lens[c_idx] - 1) <= k_lens[idx] <= k_lens[c_idx] + 1 \
                    and (keywords[c_idx] in keywords[idx] or keywords[idx] in keywords[c_idx]):

                # remove the lower scored keyword
                del keywords[idx]

            idx += 1
        c_idx += 1

    return keywords


def bm25(text, n=5, lang='dutch'):
    """
    :param text: Text that the keywords are extracted from
    :param n: Number of keywords to return
    :param lang: Language for the stopwords
    :return: Top n keywords, ordered from most to least relevant
    """
    # check model has been trained
    if words is None or freq_matrix is None:
        print("BM25 hasn't been trained! Aborting...")

    # get list of dutch stopwords
    stopwords = nltk.corpus.stopwords.words(lang)

    # get statistics for the data. the text is at position 0 in the frequency matrix
    global words
    global freq_matrix
    words, freq_matrix = fit(text, stopwords)

    # global parameters
    l_avg = np.mean(freq_matrix.sum(axis=1))
    n_docs = freq_matrix.shape[0]

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
        s = score_bm25(tf_dt, l_d, l_avg, n_docs, dft)
        scores.append((words[idx], s))

    # reorder and extract keywords
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [pair[0] for pair in scores]

    # filter keywords by removing redundancy
    keywords = remove_redundancy(keywords)

    return keywords[0:n] if len(keywords) >= n else keywords
