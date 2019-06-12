import nltk
import numpy as np
from math import log
from sklearn.feature_extraction.text import CountVectorizer

global _words
global _freq_matrix
global _vectorizer


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
    global _words
    global _freq_matrix
    global _vectorizer

    # get list of dutch stopwords
    stopwords = nltk.corpus.stopwords.words(lang)

    # get word list and frequency matrix (an ndarray where rows=documents and cols=words)
    _vectorizer = CountVectorizer(stop_words=stopwords,
                                  strip_accents='unicode',
                                  ngram_range=(1, 3))
    freq_matrix = _vectorizer.fit_transform(dataset)

    # store trained parameters as global variables

    _words = _vectorizer.get_feature_names()
    _freq_matrix = freq_matrix.toarray()


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
                    and (keywords[c_idx] in keywords[idx] or keywords[idx] in keywords[c_idx]):  # fixme
                # remove the lower scored keyword
                del keywords[idx]

            idx += 1
        c_idx += 1

    return keywords


# TODO: this is way too slow
def test(text, arguments, n=-1, lang='dutch'):
    """
    :param text: Text that the keywords are extracted from
    :param n: Number of keywords to return. Use -1 to return all of them
    :param lang: Language for the stopwords
    :return: Top n keywords, ordered from most to least relevant
    """
    global _words
    global _freq_matrix
    global _vectorizer

    # check model has been trained
    if _words is None or _freq_matrix is None:
        print("[WARNING] BM25 hasn't been trained! Returning nothing...")
        return []

    # get frequency matrix for the text
    text_freq_matrix = _vectorizer.fit_transform([text])
    text_words = _vectorizer.get_feature_names()
    text_freq_matrix = text_freq_matrix.toarray()

    # the text is at position 0 in the frequency matrix
    _freq_matrix = np.vstack((np.zeros((1, len(_words))), _freq_matrix))

    # update word counts given any old or new words in text_words
    for w in text_words:

        height, width = _freq_matrix.shape

        if w in _words:
            _freq_matrix[0, _words.index(w)] += text_freq_matrix[0, text_words.index(w)]
        else:
            _words.append(w)

            # add one column to the matrix
            new_matrix = np.zeros((height, width + 1))
            new_matrix[:, :, -1] = _freq_matrix
            _freq_matrix = new_matrix

            _freq_matrix[0, -1] += text_freq_matrix[0, text_words.index(w)]

    # global parameters
    l_avg = np.mean(_freq_matrix.sum(axis=1))
    n_docs = _freq_matrix.shape[0]

    # document parameters
    l_d = sum(_freq_matrix[0])

    # indices of the possible document keywords
    keyword_idxs = np.nonzero(_freq_matrix[0])[0].tolist()

    # compute bm25 scores
    scores = []
    for idx in keyword_idxs:
        # term level
        tf_dt = _freq_matrix[0, idx]
        dft = np.count_nonzero(_freq_matrix[:, idx])
        s = score_bm25(tf_dt, l_d, l_avg, n_docs, dft)
        scores.append((_words[idx], s))

    # reorder and extract keywords
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [pair[0] for pair in scores]

    # filter keywords by removing redundancy
    # TODO: fix the remove_redundancy function
    # keywords = remove_redundancy(keywords)

    return keywords[0:n] if len(keywords) >= n else keywords
