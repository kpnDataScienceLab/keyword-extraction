import pke 
import nltk
import string
from pke import compute_document_frequency

"""
from git_pke import pke_textrank


"""




def returnKeywords(topNkeyphrases):
	output = []
	for phrase in topNkeyphrases:
		output.append(phrase[0])
	return output

# TextRank [tested]
def pke_textRank(text, n = 5, language = 'nl'):
	
	textRank_extractor = pke.unsupervised.TextRank()
	POS = {'NOUN', 'PROPN', 'ADJ'}
	textRank_extractor.load_document(text, 
		language = language,
		normalization = None)
	textRank_extractor.candidate_weighting(window = 2,
											pos = POS,
											top_percent = 0.35)
	keyphrases = textRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

# [tested]
def test(text, arguments, n=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_textRank(text, n = n, language = 'nl')
	else:
		return pke_textRank(text, n = n, language = 'en')