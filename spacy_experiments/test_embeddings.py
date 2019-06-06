import pandas as pd 
from loadCustomDutchW2V import word2VecToSpacy
from spacy.lang.nl.stop_words import STOP_WORDS
import spacy

"""
This file is used to test different word2vec embeddings
found online.

"""


def csvToTranscripts(filename = 'aligned_epg_transcriptions_npo1_npo2.csv'):

	data = pd.read_csv(filename)

	# get list of texts
	texts = data['text']
	channels = data['channel']

	return texts, channels


def main():
	# 1. Read transcripts
	transcripts, _ = csvToTranscripts("../../aligned_epg_transcriptions_npo1_npo2.csv")

	# 2. Remove all stopwords


	# 3. Test different spacy models

	# dutch_default = spacy.load("nl_core_news_sm")

	# This one takes about 4 minutes to load.
	# nlp_nordic_dutch = word2VecToSpacy( 
	# 	"../../Word2VecModels_Dutch/Dutch_Word2Vec.zip",
	# 	"Dutch_Word2Vec/model.txt")

	# This takes about 1 minute to load. 
	cow_small_dutch = word2VecToSpacy(
		"../../Word2VecModels_Dutch/320.zip",
		"320/cow-320.txt")

if __name__ == '__main__':
	main()





