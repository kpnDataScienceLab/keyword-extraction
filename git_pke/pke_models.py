import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import pke 
from preprocessing.cleandata import readCleanTranscript

# ---- Globals used for models -----
# Yake:
yake_extractor = pke.unsupervised.YAKE()

# TfIdf requires training ONCE:
train_tfidf = True
tfidf = pke.unsupervised.TfIdf()


# YAKE
def pke_yake(text, n = 5, language = 'nl'):

	yake_extractor.load_document(text, language = language)
	yake_extractor.candidate_selection()
	yake_extractor.candidate_weighting()

	keyphrases = yake_extractor.get_n_best(n = 5)
	output = []
	for phrase in keyphrases:
		output.append(phrase[0])
	return output


# TfIdf
def pke_tfidf(text, n = 5, language = 'nl'):
	pass


# KPMiner
def pke_kpminer(text, n = 5, language = 'nl'):
	pass


# TextRank
def pke_textRank(text, n = 5, language = 'nl'):
	pass


# SingleRank
def pke_singleRank(text, n = 5, language = 'nl'):
	pass


# TopicRank
def pke_topicRank(text, n = 5, language = 'nl'):
	pass


# TopicalPageRank
def pke_topicalPageRank(text, n = 5, language = 'nl'):
	pass


# PositionRank
def pke_positionRank(text, n = 5, language = 'nl'):
	pass


# MultipartiteRank
def pke_multipartieRank(text, n = 5, language = 'nl'):
	pass


# Kea
def pke_kea(text, n = 5, language = 'nl'):
	pass


# WINGNUS
def pke_wingnus(text, n = 5, language = 'nl'):
	pass



if __name__ == '__main__':

	transcript = ' '.join(readCleanTranscript(
		"../preprocessing/clean_transcripts_june11.txt", 
		4))

	print(pke_yake(transcript, n = 10, language = 'nl'))