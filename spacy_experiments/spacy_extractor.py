import pandas as pd 
import spacy 
from spacy import displacy

def csvToTranscripts(filename = 'aligned_epg_transcriptions_npo1_npo2.csv'):

	data = pd.read_csv(filename)

	# get list of texts
	texts = data['text']
	channels = data['channel']

	return texts, channels


"""
This file is used to see how well 
we can extract keywords simply using 
the functionality of spacy.


"""

def filterNounChunks():
	pass

def filterEntities():
	pass

# Function to spot if token is just a series of punctuations
def isRepetitivePunct(token):

	# "..", "...", "....", "....."
	if (len(token) >= 2):
		for letter in token:
			if letter != '.': # <-- we will make no assumption on the data
								# meaning we assume ".kat." might appear.
				return False
		return True
	return False


"""
transcript: A single string of a transcript from a row in the .csv
dutch_nlp: A loaded spacy language object for dutch
"""
def extractSentences(transcript, dutch_nlp):
	
	sentences = []
	current_sentence = ""
	document = dutch_nlp(transcript)

	# Some useful flags
	outsideSnippets = 1
	dash = False
	startOfSentence = True
	noRepetitiveSpecials = True


	for token in document:
		# 1. Check if we are entering or leaving a snippet
		if(token.text == '\''):
			outsideSnippets *= -1 
			continue
		if outsideSnippets == 1:
			continue

		# 2. If we are within a snippet

		# --	Skip if we are in a place where subtitles are cut off
		if isRepetitivePunct(token.text):
			continue

		# --	Deal with commas
		elif(token.text in [',', ':', ';', '#', '%', '@']):
			if startOfSentence:
				continue
			if noRepetitiveSpecials == True:
				continue
			else:
				noRepetitiveSpecials = True
				current_sentence += token.text

		# -- 	Deal with hyphens (no immediate spaces)
		elif(token.text in ['-', '\'s']):
			noRepetitiveSpecials = False
			if startOfSentence:
				continue
			current_sentence += token.text
			if token.text == '-': 
				dash = True

		# --	Are we at the end of a sentence?
		elif(token.text in ['.', '!', '?']):
			noRepetitiveSpecials = False
			if startOfSentence:
				continue
			current_sentence += token.text
			result = current_sentence[1:]
			# Check sentence capitalization
			if result[0].islower():
				result[0].capitalize()
			sentences.append(result)
			current_sentence = ""
			startOfSentence = True
		# --	Add regular tokens to the sentence (normal words)
		else:
			noRepetitiveSpecials = False
			if not dash:
				if startOfSentence:
					current_sentence += ' ' + token.text.capitalize()
					startOfSentence = False
				else:
					current_sentence += ' ' + token.text
					
			else:
				current_sentence += token.text
				dash = False				
		
	return sentences






def main():
	# 1. Read transcripts
	transcripts, _ = csvToTranscripts("../../aligned_epg_transcriptions_npo1_npo2.csv")

	dutch_nlp = spacy.load("nl_core_news_sm")

	# Given the raw data, clean it up and arrange the sentences into 
	# coherent sentence strings. 
	sentence_strings = extractSentences(transcripts[200], dutch_nlp)

	



if __name__ == '__main__':
	main()