import pke 
import nltk
import string
from pke import compute_document_frequency


def returnKeywords(topNkeyphrases):
	output = []
	for phrase in topNkeyphrases:
		output.append(phrase[0])
	return output

# PositionRank
def pke_positionRank(text, n = 5, language = 'nl'):

	posRank_extractor = pke.unsupervised.PositionRank()
	
	POS = {'NOUN', 'PROPN', 'ADJ'}
	grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

	posRank_extractor.load_document(input = text,
									language = language,
									normalization = None)
	posRank_extractor.candidate_selection(grammar = grammar,
									maximum_word_number = 3)
	posRank_extractor.candidate_weighting(window = 10,
									pos = POS)
	keyphrases = posRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)	

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, n=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_positionRank(text, n = n, language = 'nl')
	else:
		return pke_positionRank(text, n = n, language = 'en')