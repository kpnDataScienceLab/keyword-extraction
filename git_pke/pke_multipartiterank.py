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


# MultipartiteRank
def pke_multipartiteRank(text, arguments, n=5, language='dutch'):
    multiPartiteRank_extractor = pke.unsupervised.MultipartiteRank()
    parser_language = 'nl' if language == 'dutch' else 'en'

    POS = {'NOUN', 'PROPN', 'ADJ'}
    multiPartiteRank_extractor.load_document(input=text, language=parser_language)
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += nltk.corpus.stopwords.words(language)
    multiPartiteRank_extractor.candidate_selection(pos=POS,
                                                   stoplist=stoplist)

    if (len(arguments) < 3):
    	alpha = 1.1
    	threshold = 0.74
    	method = 'average'
    else:
    	alpha = float(arguments[0])
    	threshold = float(arguments[1])
    	method = arguments[2]
    multiPartiteRank_extractor.candidate_weighting(alpha=alpha, # 1.1
                                                   threshold=threshold, # 0.74
                                                   method=method) # 'average'
    keyphrases = multiPartiteRank_extractor.get_n_best(n=n)
    return returnKeywords(keyphrases)


# Required for interfacing
def train(dataset, arguments, lang='dutch'):
    pass


def test(text, arguments, k=5, lang='dutch'):
    if (lang == 'dutch'):
        return pke_multipartiteRank(text, arguments, n=k, language='dutch')
    else:
        return pke_multipartiteRank(text, arguments, n=k, language='english')

