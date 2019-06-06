from spacy import displacy
import spacy
import pandas as pd 

"""
This file is used to test how well 
displacy can be used to visualize results. 


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
	dutch_nlp = spacy.load("nl_core_news_sm")

	# 3. Display entities
	displacy.serve(dutch_nlp(transcripts[0]), style = "ent")

if __name__ == '__main__':
	main()