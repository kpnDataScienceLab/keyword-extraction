import pke
import nltk
import string
from pke import compute_document_frequency

"""
Author: Arvid Lindstrom,
June 2019.
Arvid.Lindstrom@gmail.com
"""

def returnKeywords(topNkeyphrases):
    output = []
    for phrase in topNkeyphrases:
        output.append(phrase[0])
    return output


# TopicRank, takes default language as 'dutch' instead of 'nl' [tested]
def pke_topicRank(text, arguments, n=5, language='dutch'):
    topicRank_extractor = pke.unsupervised.TopicRank()

    POS = {'NOUN', 'PROPN', 'ADJ'}
    # Special characters
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += nltk.corpus.stopwords.words(language)

    if (len(arguments) < 2):
        threshold = 0.74
        method = 'average'
    else:
        threshold = float(arguments[0])
        method = arguments[1]


    topicRank_extractor.load_document(input=text)
    topicRank_extractor.candidate_selection(pos=POS,
                                            stoplist=stoplist)
    topicRank_extractor.candidate_weighting(threshold=threshold, # 0.74
                                            method=method) # 'average'
    keyphrases = topicRank_extractor.get_n_best(n=n)
    return returnKeywords(keyphrases)


# Required for interfacing
def train(dataset, arguments, lang='dutch'):
    pass


def test(text, arguments, k=5, lang='dutch'):
    if lang == 'dutch':
        return pke_topicRank(text, arguments, n=k, language='dutch')
    else:
        return pke_topicRank(text, arguments, n=k, language='english')
