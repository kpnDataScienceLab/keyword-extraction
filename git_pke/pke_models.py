import pke 
import nltk
import string
from pke import compute_document_frequency
from cleandata import readCleanTranscript
nltk.download('stopwords')


"""
Written by Arvid L. June 12th

To run anything using pke:

pip install git+https://github.com/boudinfl/pke.git

# Requirements
python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
# download the english model
python -m spacy download en 

To test models use the following interface:


from git_pke import pke_yake, pke_textRank, pke_singleRank, \
	pke_topicRank, pke_positionRank, pke_multipartieRank

## Functions are called as 
topN = pke_yake(text, n = 5, language = 'nl')
topN = pke_textRank(text, n = 5, language = 'nl')
topN = pke_singleRank(text, n = 5, language = 'nl')
topN = pke_topicRank(text, n = 5, language = 'dutch')
topN = pke_positionRank(text, n = 5, language = 'nl')
topN = pke_multipartieRank(text, n = 5, language = 'dutch')

"""



# ---- Globals used for models -----
yake_extractor = pke.unsupervised.YAKE()
textRank_extractor = pke.unsupervised.TextRank()
singleRank_extractor = pke.unsupervised.SingleRank()
topicRank_extractor = pke.unsupervised.TopicRank()
posRank_extractor = pke.unsupervised.PositionRank()
multiPartiteRank_extractor = pke.unsupervised.MultipartiteRank()

def returnKeywords(topNkeyphrases):
	output = []
	for phrase in topNkeyphrases:
		output.append(phrase[0])
	return output

# Priority models:

# YAKE [tested]
def pke_yake(text, n = 5, language = 'nl'):

	yake_extractor.load_document(text, language = language)
	yake_extractor.candidate_selection()
	yake_extractor.candidate_weighting()

	keyphrases = yake_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# TextRank [tested]
def pke_textRank(text, n = 5, language = 'nl'):
	
	POS = {'NOUN', 'PROPN', 'ADJ'}
	textRank_extractor.load_document(text, 
		language = language,
		normalization = None)
	textRank_extractor.candidate_weighting(window = 2,
											pos = POS,
											top_percent = 0.35)
	keyphrases = textRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# SingleRank [tested]
def pke_singleRank(text, n = 5, language = 'nl'):
	POS = {'NOUN', 'PROPN', 'ADJ'}

	singleRank_extractor.load_document(input = text,
										language = language,
									normalization = None)
	singleRank_extractor.candidate_selection(pos = POS)
	singleRank_extractor.candidate_weighting(window = 10,
											pos = POS)
	keyphrases = singleRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# TopicRank, takes default language as 'dutch' instead of 'nl' [tested]
def pke_topicRank(text, n = 5, language = 'dutch'):
	
	POS = {'NOUN', 'PROPN', 'ADJ'}
	# Special characters
	stoplist = list(string.punctuation)
	stoplist += ['-lrb-', '-rrb', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	stoplist += nltk.corpus.stopwords.words(language)

	topicRank_extractor.load_document(input = text)
	topicRank_extractor.candidate_selection(pos = POS, 
										stoplist = stoplist)
	topicRank_extractor.candidate_weighting(threshold = 0.74,
											method = 'average')
	keyphrases = topicRank_extractor.get_n_best(n = n)
	return returnKeywords(keyphrases)

# PositionRank
def pke_positionRank(text, n = 5, language = 'nl'):
	
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

# MultipartiteRank
def pke_multipartieRank(text, n = 5, language = 'dutch'):
	
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




# WINGNUS
def pke_wingnus(text, n = 5, language = 'nl'):
	pass





# Low priority models

# TfIdf
def pke_tfidf(text, n = 5, language = 'nl'):
	pass

# KPMiner
def pke_kpminer(text, n = 5, language = 'nl'):
	pass




if __name__ == '__main__':

	transcript = ' '.join(readCleanTranscript(
		"clean_transcripts_june11.txt", 
		4))

	print("Running yake:")
	print(pke_yake(transcript, n = 10, language = 'nl'))
	print("Running textRank:")
	print(pke_textRank(transcript, n = 10, language = 'nl'))
	print("Running singleRank:")
	print(pke_singleRank(transcript, n = 10, language = 'nl'))
	print("Running topicRank:")
	print(pke_topicRank(transcript, n = 10, language = 'dutch'))
	print("Running positionRank:")
	print(pke_positionRank(transcript, n = 10, language = 'nl'))
	print("Running MultipartiteRank:")
	print(pke_multipartieRank(transcript, n = 10, language = 'dutch'))
