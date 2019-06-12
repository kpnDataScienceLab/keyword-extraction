import pke 
import nltk
import string
from pke import compute_document_frequency
# from cleandata import readCleanTranscript
nltk.download('stopwords')
import json

# Comment out following code before pushing
# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 

# from eval_metrics import average_precision
# from datasets.datasets import Dataset
# -----------------------------------------end


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


from git_pke.pke_models import pke_yake, pke_textRank, pke_singleRank, \
	pke_topicRank, pke_positionRank, pke_multipartieRank

## Functions are called as, 
topN = pke_yake(text, n = 5, language = 'nl') # or 'en'
topN = pke_textRank(text, n = 5, language = 'nl')
topN = pke_singleRank(text, n = 5, language = 'nl')
topN = pke_topicRank(text, n = 5, language = 'dutch') # or 'english'
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

	# Use this for Dutch testing on TV data
	# transcript = ' '.join(readCleanTranscript(
	# 	"clean_transcripts_june11.txt", 
	# 	4))

	data = Dataset("500N-KPCrowd")

	path = "../datasets/ake-datasets/datasets/500N-KPCrowd/train/"
	filename = "art_and_culture-20893614.xml"
	path = path + filename

	transcript = data.parse_xml(path)
	language1 = 'en'		# nl
	language2 = 'english' 	# dutch
	print("Input:\n\n")
	print(transcript)

	labels_path = "../datasets/ake-datasets/datasets/500N-KPCrowd/references/"
	lfile = "train.reader.json"
	with open(labels_path + lfile) as handle:
		l = json.load(handle)	
	labels = l[filename[:-4]]
	for idx in range(len(labels)):
		labels[idx] = labels[idx][0]


	print("\n\nPredictions: ")
	print("Running yake:")
	print(pke_yake(path, n = 10, language = language1))
	print("Running textRank:")
	print(pke_textRank(path, n = 10, language = language1))
	print("Running singleRank:")
	print(pke_singleRank(path, n = 10, language = language1))
	print("Running topicRank:")
	print(pke_topicRank(path, n = 10, language = language2))
	print("Running positionRank:")
	print(pke_positionRank(path, n = 10, language = language1))
	print("Running MultipartiteRank:")
	print(pke_multipartieRank(path, n = 10, language = language2))

	print("\n\nLabels: ", labels)

	print("Scores:")
