import spacy, nl_core_news_sm
from cleandata import *
from spacy import displacy
import argparse

"""

File candidate keywords works as follows:
from preprocessing.candidatekeywords import candidateKeywords

# 1. 
dutch_nlp = spacy.load("nl_core_news_sm")

# 2. 



"""




def filterNounChunks(nlp_document):
	
	# for token in nlp_document:
	# 	print(token.text, token.pos_, token.dep_, ' <---- ' , token.head.text, token.head.pos_,
	# 		[child for child in token.children])
	# print("-------------RESULT------------")

	noun_chunks = []
	current_chunk = ""
	child_chunk = ""
	for token in nlp_document:
		if token.pos_ == "NOUN":
			for child in token.children:
				if child.pos_ == "ADJ":
					current_chunk += (child.text + ' ')
				if child.pos_ == "NOUN":
					for grandchild in child.children:
						child_chunk += (grandchild.text + ' ')
					child_chunk += child.text + ' '
			current_chunk += (token.text + ' ' + child_chunk)
			if(current_chunk[-1] == ' '):
				current_chunk = current_chunk[:-1]
			noun_chunks.append(current_chunk)
			current_chunk = ""
			child_chunk = ""

	return noun_chunks

def removeDuplicates(candidate_list):
	return list(set(candidate_list))

def simpleNouns(nlp_document):
	nouns = []
	for token in nlp_document:
		if token.pos_ == "NOUN":
			nouns.append(token.text)
	return nouns 

def filterEntities(nlp_document):
	entities = []
	for token in nlp_document.ents:
		entities.append(token.text)
	return entities

"""
Function candidateKeywords takes as input the index 
of the transcript we are interested in and a loaded 
spacy model for dutch "nl_core_news_sm". 
Returns a list of candidate keywords based on 
entities and noun-phrases. 

"""
def candidateKeywords(tranIndex, dutch_nlp, input_filename = "clean_week_1_transcripts.txt"):

	cleanTranscript = readCleanTranscript(input_filename, tranIndex)
	candidates = []
	for cleanTrans in cleanTranscript:
		doc = dutch_nlp(cleanTrans)
		candidates += filterEntities(doc)
		candidates += filterNounChunks(doc)
		candidates += simpleNouns(doc)

	candidates = removeDuplicates(candidates)

	return candidates


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--idx', default=0, type=int, help='which sentence')
	parser.add_argument('--trans', default=0, type=int, help='which transcript')
	ARGS = parser.parse_args()

	# 1. Read transcripts
	input_filename = "clean_week_1_transcripts.txt"

	# 2. Load spacy model
	dutch_nlp = spacy.load("nl_core_news_sm")

	print("---------INPUT----------")
	print(readCleanTranscript(input_filename, ARGS.trans))
	print("---------OUTPUT---------")
	print(candidateKeywords(ARGS.trans, dutch_nlp, input_filename))


	# joseph = "Voor het grote huis van mijn grote ouders ligt een kleine hond"
	# print(joseph)
	# print(filterNounChunks(dutch_nlp(joseph)))

if __name__ == '__main__':
	main()