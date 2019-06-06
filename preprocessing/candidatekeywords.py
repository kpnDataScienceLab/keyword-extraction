import spacy
from cleandata import *


def filterNounChunks():
	pass

def filterEntities():
	pass







def main():

	# 1. Read transcripts
	transcripts, _ = csvToTranscripts("../../aligned_epg_transcriptions_npo1_npo2.csv")

	dutch_nlp = spacy.load("nl_core_news_sm")



if __name__ == '__main__':
	main()