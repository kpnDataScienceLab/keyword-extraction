import pandas as pd 
import spacy 

"""
This file exists to ensure that the 
text is processed in a format that 
preserves sentence structure. This is 
required on the TV-data since the 
sentences are split into 'snippets'.
Motivation: Spacy can make use of 
sophisticated sentence parsing if 
the sentences arrive properly. 
"""




def csvToTranscripts(filename = 'aligned_epg_transcriptions_npo1_npo2.csv'):

	data = pd.read_csv(filename)

	# get list of texts
	texts = data['text']
	channels = data['index'] #<-- can be used to see which TV show text came from 

	return texts, channels


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

"""
Provided that the file clean_week_1_transcripts.txt has been generated, 
this function will take the path to the file and the desired 
index of the transcript we want and return it as a list of 
strings.
"""
def readCleanTranscript(input_path, transcriptIdx):

	# 1. Find the line to start reading from 
	file = open(input_path, "r")
	lines = file.readlines()

	start_index = [x for x in range(len(lines)) if \
		("#clean_transcript#" + str(transcriptIdx)) in lines[x]][0]
	end_index = [x for x in range(len(lines)) if \
		(("#clean_transcript#" + str(transcriptIdx + 1)) in lines[x])][0]

	clean_sentences = []
	for sentence in lines[start_index + 1 : end_index]:
		clean_sentences.append(sentence.rstrip())

	file.close()
	return clean_sentences

	



def saveTranscriptsToFile(input_path, output_path):

	file = open(output_path, "w+")

	# 1. Read transcripts
	transcripts, _ = csvToTranscripts(input_path)

	dutch_nlp = spacy.load("nl_core_news_sm")

	# Given the raw data, clean it up and arrange the sentences into 
	# coherent sentence strings. 
	for idx in range(len(transcripts)):
		print("Cleaning transcript ", idx)
		file.write("#clean_transcript#" + str(idx) + '\n')
		clean_transcript = extractSentences(transcripts[idx], dutch_nlp)
		for sentence in clean_transcript:
			file.write(sentence + '\n')

	file.close()

if __name__ == '__main__':

	output_filename = "clean_transcripts_june11.txt"
	input_path = "../transcriptions.csv"

	# ------------- Uncomment this line to generate and save the clean transcripts
	# saveTranscriptsToFile(input_path, output_filename)
	# ----------------------------------------------------------------------------
	
	transcripts, _ = csvToTranscripts(input_path)

	# This is an example of how to access transcripts from now on. 
	# Note that this is NOT the proposed candidate keywords, this is the 
	# data used by spacy to generate those keywords. 
	cleanTranscript4 = readCleanTranscript(output_filename, 4)
	
	print("Input transcript ----------------------\n")
	print(transcripts[4])
	print("\nCleaned up transcript -----------------\n")
	print(cleanTranscript4)