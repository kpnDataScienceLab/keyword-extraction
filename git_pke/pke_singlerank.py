import pke 
import nltk
import string
from pke import compute_document_frequency


def returnKeywords(topNkeyphrases):
	output = []
	for phrase in topNkeyphrases:
		output.append(phrase[0])
	return output

# SingleRank [tested]
def pke_singleRank(text, n = 5, language = 'nl'):

	singleRank_extractor = pke.unsupervised.SingleRank()

	POS = {'NOUN', 'PROPN', 'ADJ'}

	singleRank_extractor.load_document(input = text,
										language = language,
									normalization = None)
	singleRank_extractor.candidate_selection(pos = POS)
	singleRank_extractor.candidate_weighting(window = 10,
											pos = POS)
	keyphrases = singleRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

# [tested]
def test(text, arguments, n=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_singleRank(text, n = n, language = 'nl')
	else:
		return pke_singleRank(text, n = n, language = 'en')