from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.nl.stop_words import STOP_WORDS
import pandas as pd
import zipfile
import gensim
import warnings

warnings.filterwarnings("ignore")


class GraphWord2Vec:
    def __init__(self, nlp, stopwords, d=0.85):
        self.d = d  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.nlp = nlp
        self.stopwords = stopwords

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in self.stopwords:
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        chunks = []
        # print(type(doc.noun_chunks))
        # print(type(doc.ents))

        for chunk in doc.noun_chunks:
            if lower is True:
                selected_words = (self.nlp(chunk.text.lower()))
            else:
                selected_words = chunk
            chunks.append(selected_words)
        return chunks

    def get_vocab(self, chunks):
        """Get all tokens"""
        vocab = OrderedDict()
        check = set()
        i = 0
        for chunk in chunks:
            if chunk.text not in check:
                vocab[chunk] = i
                i += 1
                check.add(chunk.text)
        return vocab

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1 in vocab:
            for word2 in vocab:
                i, j = vocab[word1], vocab[word2]
                g[i][j] = word1.similarity(word2)
                # print(f"{word1} {word2} -- {g[i][j]}")

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        return [k.text for k in list(node_weight.keys())[:number]]

        # for i, (key, value) in enumerate(node_weight.items()):
        #     print(str(key) + ' - ' + str(value))
        #     if i > number:
        #         break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'], lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Parse text by spaCy
        doc = self.nlp(text)

        # Filter sentences
        sentences = self.segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for step in range(self.steps):

            pr = (1 - self.d) + self.d * np.dot(g, pr)
            diff = sum(abs(previous_pr - pr))

            # print(f"step: {step}, diff:{diff}")
            if diff < self.min_diff:
                break
            else:
                previous_pr = pr

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        self.node_weight = node_weight


global _model, loaded
_model = None
loaded = False


def train(dataset, arguments, lang='dutch'):
    global loaded
    if loaded:
        return
    loaded = True
    global _model
    if lang == 'dutch':
        from spacy.lang.nl.stop_words import STOP_WORDS
        model_data_path = "39.zip"
        with zipfile.ZipFile(model_data_path, "r") as archive:
            stream = archive.open("model.txt")

        # 1. How to load a model from the Nordic Language Processing Laboratory
        model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=False, unicode_errors='replace')

        nlp = spacy.load("nl_core_news_sm")
        keys = []
        for idx in range(len(model.vocab)):
            keys.append(model.index2word[idx])
        nlp.vocab.vectors = spacy.vocab.Vectors(data=model.syn0, keys=keys)

    if lang == 'english':
        nlp = spacy.load("en_core_web_lg")
        from spacy.lang.en.stop_words import STOP_WORDS

    _model = GraphWord2Vec(nlp, STOP_WORDS, *arguments)


def test(text, arguments, k=5, lang=5):
    global _model
    _model.analyze(text, candidate_pos=['NOUN', 'PROPN'], lower=False)
    return _model.get_keywords(k)


def get_model():
    global _model
    return _model
