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

# SingleRank [tested]
def pke_singleRank(text, arguments, n = 5, language = 'nl'):

	singleRank_extractor = pke.unsupervised.SingleRank()

	POS = {'NOUN', 'PROPN', 'ADJ'}

	if (len(arguments) == 0):
		window = 10
	else:
		window = int(arguments[0])

	singleRank_extractor.load_document(input = text,
										language = language,
									normalization = None)
	singleRank_extractor.candidate_selection(pos = POS)
	singleRank_extractor.candidate_weighting(window = window, # 10
											pos = POS)
	keyphrases = singleRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, k=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_singleRank(text, arguments, n = k, language = 'nl')
	else:
		return pke_singleRank(text, arguments, n = k, language = 'en')