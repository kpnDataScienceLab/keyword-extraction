import pandas as pd
import pickle as pkl
import re


def get_texts(fname):
    """
    Returns a list of cleaned up texts (i.e. removed inverted commas
    and commas) coming from the file fname
    :param fname: the path of the csv file containing the text files
        in column 'text'
    :return: a list of strings where each string is a text
    """
    ts = pd.read_csv(fname)["text"]
    return ts


def get_stopwords(size):
    """
    Returns a list containing the medium stopword list (~260 words) or the large
    stopword list (~500 words)
    :param size: 'medium' or 'large'
    :return: a list containing all stopwords
    """
    fname = "stopwords/stopwords_" + size + ".pkl"
    with open(fname, "rb") as handle:
        stopwords = pkl.load(handle)
    return stopwords
