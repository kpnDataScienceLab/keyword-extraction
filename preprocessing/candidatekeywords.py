import spacy
from cleandata import *
from spacy import displacy



def filterNounChunks(nlp_document):
	
	for token in nlp_document:
		print(token.text, token.pos_, token.dep_, ' <==== ' , token.head.text, token.head.pos_,
			[child for child in token.children])

def filterEntities(nlp_document):
	for token in nlp_document.ents:
		print(token.text)


def candidateKeywords(cleanTranscript):

	pass






def main():

	# 1. Read transcripts
	input_filename = "clean_week_1_transcripts.txt"
	transcripts, _ = csvToTranscripts("../../aligned_epg_transcriptions_npo1_npo2.csv")
	clean_transcript45 = readCleanTranscript(input_filename, 45)

	# 2. Load spacy model
	dutch_nlp = spacy.load("nl_core_news_sm")

	# 3. Filter noun chunks
	print(clean_transcript45[1])
	filterNounChunks(dutch_nlp(clean_transcript45[1]))

	# 4. Filter entities
	filterEntities(dutch_nlp(clean_transcript45[1]))

	# 5. Display dependency tree
	displacy.serve(dutch_nlp(clean_transcript45[1]), style="dep")


if __name__ == '__main__':
	main()