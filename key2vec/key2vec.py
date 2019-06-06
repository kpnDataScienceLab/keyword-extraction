import string
import re
import nltk
import pickle
import numpy as np
from scipy.spatial.distance import cosine

with open('embeddings/fasttext_embeddings.pkl', 'rb') as handle:
    embeddings = pickle.load(handle)


def fasttext(word):
    return np.array(embeddings.get(word, [0.] * 300))


def clean_text(text):
    # remove punctuation except for dashes
    punctuation = string.punctuation.replace('-', '')
    text = text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))

    # remove leading or trailing dashes
    text = re.sub(r'- | -', ' ', text)

    # remove duplicate spaces
    text = re.sub(r' +', ' ', text)

    # make text lowercase
    return text.lower()


def get_candidate_keywords(text):
    # get all words in the text
    candidates = text.split()

    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('dutch')
    return list(set(candidates).difference(set(stopwords)))


def get_topic_vector(topic_description):
    """
    Given a text containing a summary description of the tv program, it returns a vector
    embedding for that description.
    :param topic_description: A summary for the text under consideration
    :return: An embedding for the summary
    """
    topic_description = clean_text(topic_description)
    words = topic_description.split()
    return np.mean([fasttext(w) for w in words], axis=0)


def key2vec(text, topic_description, n=5):
    """
    :param text: Text that the keywords are extracted from
    :param topic_description: Text that should summarize the topic of the text
    :param n: Number of keywords to return
    :return: Top n keywords, ordered from most to least relevant
    """

    # ----- CANDIDATE SELECTION -----

    # produce candidate keywords and clean the text
    text = clean_text(text)
    candidates = get_candidate_keywords(text)

    # ------ CANDIDATE SCORING ------

    # get vector for the topic of the document
    topic_vector = get_topic_vector(topic_description)

    # cosine similarity measures for rescaling later
    min_cossim = 1
    max_cossim = 0

    # get the keyword embeddings and their similarity to the topic
    word_dicts = []
    for candidate in candidates:
        words = candidate.split()
        if len(words) > 1:
            vector = np.mean([fasttext(w) for w in words], axis=0)
        else:
            vector = fasttext(candidate)

        # compute cosine(keyword, topic)
        cossim = cosine(vector, topic_vector)
        if cossim > max_cossim:
            max_cossim = cossim
        if cossim < min_cossim:
            min_cossim = cossim

        word_dicts.append({'keyword': candidate, 'embedding': vector, 'cosine': cossim})

    # rescale cosine similarities
    for word_dict in word_dicts:
        word_dict['cosine'] = (word_dict['cosine'] - min_cossim) / (max_cossim - min_cossim)

    # ------ CANDIDATE RANKING ------

    # TODO: implement candidate ranking
    raise NotImplementedError

    keywords = []

    return keywords[0:n] if len(keywords) >= n else keywords
