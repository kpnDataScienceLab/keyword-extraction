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

# PositionRank
def pke_positionRank(text, arguments, n = 5, language = 'nl'):

	posRank_extractor = pke.unsupervised.PositionRank()
	
	POS = {'NOUN', 'PROPN', 'ADJ'}
	grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

	if (len(arguments) < 2):
		maximum_word_number = 3
		window = 10
	else:
		maximum_word_number = int(arguments[0])
		window = int(arguments[1])

	posRank_extractor.load_document(input = text,
									language = language,
									normalization = None)
	posRank_extractor.candidate_selection(grammar = grammar,
									maximum_word_number = maximum_word_number) # 3
	posRank_extractor.candidate_weighting(window = window, # 10
									pos = POS)
	keyphrases = posRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)	

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, k=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_positionRank(text, arguments, n = k, language = 'nl')
	else:
		return pke_positionRank(text, arguments, n = k, language = 'en')