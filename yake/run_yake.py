"""
Yake, as implemented by the authors of the original paper.
Requirements:
	python3
	pandas
	nltk
	files: get_data.py, yake_test.py

How to run (instructions from https://github.com/LIAAD/yake):
	
	1. Install everything using pip:
		pip install git+https://github.com/LIAAD/yake

	2. Run as: (to use default parameters)
		python run_yake.py 

Written by:
	Arvid L. June 2019. 

"""
import argparse
from yake_test import yakeTest
from yake_by_us import transcriptsToKeywords
from utils import csvToTranscripts, keywordsTocsv

def yake_keywords(text, n = 5):
	parser = argparse.ArgumentParser()
	parser.add_argument('--use_subset', default=False, type = bool, help = "Run tests on subset of 20 transcripts. ")
	parser.add_argument('--subsetSize', default=10, type = int, help = "Size of subset for testing yake.")
	parser.add_argument('--language', default="nl", 
		type=str, help = 'In what language are the input transcripts?')
	parser.add_argument('--top', default=10, type = int, help = "Number of top-relevant keywords to return")
	parser.add_argument('--n', default=n, type = int, help = "How many words a keyword can consist of")
	ARGS = parser.parse_args()

	transcripts = []
	transcripts.append(text)
	keywords_per_transcript = transcriptsToKeywords(transcripts, ["---"], ARGS)
		
	output = []
	for keyword in keywords_per_transcript[0]:
		print(keyword)
		output.append(keyword[0]) 

	return output





def main():

	if ARGS.test:
		yakeTest()
		
	elif ARGS.csv:
		transcripts, channels = csvToTranscripts(ARGS.csv_input_file)

		# 1. Finds keywords
		keywords_per_transcript = transcriptsToKeywords(transcripts, channels, ARGS)

		# 2. Produce output file
		keywordsTocsv(keywords_per_transcript, ARGS.csv_output_file)

	else:
		print("Please set either '--test True' or '--csv True' (omit apostrophes ' ')")


if __name__ == "__main__":

	csv_file_path = "../../aligned_epg_transcriptions_npo1_npo2.csv"
	csv_keyword_output = "yakeOutput"

	parser = argparse.ArgumentParser()
	# Main arguments
	parser.add_argument('--test', default=False, type=bool, help='Run yake on dummy text')
	parser.add_argument('--csv', default=True, type=bool, help = 'Run yake on testing 298 row csv')
	parser.add_argument('--csv_input_file', default = csv_file_path, type = str, help = "Name of csv file to use as input.")
	parser.add_argument('--csv_output_file', default = csv_keyword_output, type = str, help = "Name of csv file to use as output.")

	# YAKE Arguments
	parser.add_argument('--use_subset', default=False, type = bool, help = "Run tests on subset of 20 transcripts. ")
	parser.add_argument('--subsetSize', default=10, type = int, help = "Size of subset for testing yake.")
	parser.add_argument('--language', default="nl", 
		type=str, help = 'In what language are the input transcripts?')
	parser.add_argument('--top', default=10, type = int, help = "Number of top-relevant keywords to return")
	parser.add_argument('--n', default=3, type = int, help = "How many words a keyword can consist of")

	# Collect arguments into parser object ARGS
	ARGS = parser.parse_args()

	# print(ARGS.use_subset)

	# main()

	print(yake_keywords("Het was een zware ochtend. Om 7 uur ging de wekker die mij met blauwe ogen het bed uit lokte. Eenmaal aangekomen op werk werd ik door een dubbele espresso verrast.", n = 3))