import spacy, nl_core_news_sm
from cleandata import *
from spacy import displacy
import argparse
from spacy.lang.nl.stop_words import STOP_WORDS

"""

File candidate keywords works as follows:
from preprocessing.candidatekeywords import candidateKeywords

# 1. 
dutch_nlp = spacy.load("nl_core_news_sm")
english_nlp = spacy.load("en_core_web_sm")

# 2. 
index = 42
input_path = "clean_week_1_transcripts.txt"
candidates = candidateKeywords(index, dutch_nlp, input_path,
								useExistFilter = True, k = 5)
where...
"""

def isNounChunkInText(chunk_candidate, raw_text):
	return chunk_candidate in raw_text
	
def filterNounChunks(nlp_document, raw_text, useExistFilter = True):
	
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
			# Only add candidates that actually occured. 
			# (As a safety measure for difficult phrases)
			if useExistFilter:
				if isNounChunkInText(current_chunk, raw_text):
					noun_chunks.append(current_chunk)
			else:
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
Function removes keyphrases containing special symbols. 

NOTE: We allow & and @ to appear within keywords!

"""
def removeSpecials(candidate_list):
	for phrase in candidate_list:
		# Only check for letters within the keyphrase
		# (not first or last)
		for letter in range(1, len(phrase) - 1):
			
			if phrase[letter] in ['!', '#', '$', '%', \
									'^', '*', '(', ')', 
									'{', '}', '|', '~', '$', ',',
									'?']:
				candidate_list.remove(phrase)
				break
	return candidate_list

"""
Function removes kephrases of length 1
"""
def remove1letterWords(candidate_list):
	for phrase in candidate_list:
		if len(phrase) == 1:
			candidate_list.remove(phrase)
	return candidate_list

"""
Function removes keyphrases of length longer than k = 5
"""
def removeKeywordsLongerThanK(candidate_list, k = 5):
	for keyphrase in candidate_list:
		if (len(keyphrase.split()) > k):
			candidate_list.remove(keyphrase)
	return candidate_list

"""
Remove all stopwords defined by spacy for dutch
"""
def removeStopWords(candidate_list):

	for keyword in candidate_list:
		if keyword.lower() in STOP_WORDS:
			candidate_list.remove(keyword)
	return candidate_list

	# return list(set(candidate_list) - set(STOP_WORDS))

"""
Function candidateKeywords takes as input the index 
of the transcript we are interested in and a loaded 
spacy model for dutch "nl_core_news_sm". 
Returns a list of candidate keywords based on 
entities and noun-phrases. 

"""
def candidateKeywords(tranIndex, dutch_nlp, 
						input_filename = "clean_week_1_transcripts.txt",
						useExistFilter = True, k = 5):

	cleanTranscript = ' '.join(readCleanTranscript(input_filename, tranIndex))
	candidates = []
	
	# Process all sentences as one big text
	doc = dutch_nlp(cleanTranscript)
	# Find entities, noun-phrases and simple nouns
	candidates += filterEntities(doc)
	candidates += filterNounChunks(doc, cleanTranscript, 
					useExistFilter = useExistFilter)
	candidates += simpleNouns(doc)

	return finalFilter(candidates, k = k)

def finalFilter(candidate_list, k = 5):

	# Remove duplicates
	print("Before duplicate removal: ", len(candidate_list))
	candidate_list = removeDuplicates(candidate_list)
	print("After duplicate removal: ", len(candidate_list))

	# Remove any remaining stopwords
	candidate_list = removeStopWords(candidate_list)
	print("After stopword removal: ", len(candidate_list))

	# Remove candidate_list that are too long
	candidate_list = removeKeywordsLongerThanK(candidate_list, k = k)
	print("After length filtering k = 5: ", len(candidate_list))

	# Remove keywords with special characters such as "Madag%ascar"
	candidate_list = removeSpecials(candidate_list)
	print("After filtering specials: ", len(candidate_list))

	# Remove keywords with only 1 letter
	candidate_list = remove1letterWords(candidate_list)
	print("After removing \"1 letters\": ", len(candidate_list))


	return candidate_list

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--idx', default=0, type=int, help='which sentence')
	parser.add_argument('--trans', default=0, type=int, help='which transcript')
	parser.add_argument('--filename', default = "clean_transcripts_june11.txt", 
						type = str, help = "Name of input file (clean transcript)")
	parser.add_argument('--language', default = 'nl', type = str, 
						help = "In which language to extract keywords")
	ARGS = parser.parse_args()

	# 1. Read transcripts
	input_filename = ARGS.filename

	# 2. Load spacy model
	if ARGS.language == 'nl':
		nlp = spacy.load("nl_core_news_sm")
	elif ARGS.language == 'en':
		nlp = spacy.load("en_core_web_sm")
	else:
		print("Please enter valid language \n> --language nl \n> --language en")
		return 

	# print("---------INPUT (cleaned transcript)----------\n")
	print(' '.join(readCleanTranscript(input_filename, ARGS.trans)))
	print("\n---------OUTPUT (Candidate Keywords)---------\n")
	keyphrases = candidateKeywords(ARGS.trans, 
									nlp, 
									input_filename,
									useExistFilter = True,
									k = 5)

	# This final filter removes any keyphrases with special 
	# characters in them, length above a threshold k or keywords 
	# that contain only a single letter
	# keyphrases = finalFilter(keyphrases, k = 5)

	for key in keyphrases:
		print(key)

	# joseph = "Voor het grote huis van mijn grote ouders ligt een kleine hond"

if __name__ == '__main__':
	main()