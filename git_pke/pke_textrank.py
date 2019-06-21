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

# TextRank [tested]
def pke_textRank(text, arguments, n = 5, language = 'nl'):
	
	textRank_extractor = pke.unsupervised.TextRank()
	POS = {'NOUN', 'PROPN', 'ADJ'}
	textRank_extractor.load_document(text, 
		language = language,
		normalization = None)

	if (len(arguments) < 2):
		window = 2
		top_percent = 0.35
	else:
		window = int(arguments[0])
		top_percent = float(arguments[1])
	
	textRank_extractor.candidate_weighting(window = window, # 2
											pos = POS,
											top_percent = top_percent) # 0.35
	keyphrases = textRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, k=5, lang = 'dutch'):
	if(lang == 'dutch'):
		return pke_textRank(text, arguments, n = k, language = 'nl')
	else:
		return pke_textRank(text, arguments, n = k, language = 'en')