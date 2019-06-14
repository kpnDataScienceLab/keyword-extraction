import pke 
import nltk
import string
from pke import compute_document_frequency


def returnKeywords(topNkeyphrases):
	output = []
	for phrase in topNkeyphrases:
		output.append(phrase[0])
	return output

# MultipartiteRank
def pke_multipartiteRank(text, n = 5, language = 'dutch'):

	multiPartiteRank_extractor = pke.unsupervised.MultipartiteRank()
	
	POS = {'NOUN', 'PROPN', 'ADJ'}
	multiPartiteRank_extractor.load_document(input = text)
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += nltk.corpus.stopwords.words(language)
	multiPartiteRank_extractor.candidate_selection(pos = POS,
												stoplist = stoplist)
	multiPartiteRank_extractor.candidate_weighting(alpha = 1.1,
												threshold = 0.74,
												method = 'average')
	keyphrases = multiPartiteRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)


# Required for interfacing
def train(dataset,arguments,lang='dutch'):
	pass

def test(text, arguments, k=5, lang = 'dutch'):
	print(arguments)
	if(lang == 'dutch'):
		return pke_multipartiteRank(text, n = k, language = 'dutch')
	else:
		return pke_multipartiteRank(text, n = k, language = 'english')