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

def pke_yake(text, n = 5, language = 'nl'):

	yake_extractor = pke.unsupervised.YAKE()
	yake_extractor.load_document(text, language = language)
	yake_extractor.candidate_selection()
	yake_extractor.candidate_weighting()

	keyphrases = yake_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, k=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_yake(text, n = k, language = 'nl')
	else:
		return pke_yake(text, n = k, language = 'en')
