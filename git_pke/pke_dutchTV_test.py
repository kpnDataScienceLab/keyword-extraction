
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from our_yake.run_yake import yake_keywords
from preprocessing.cleandata import readCleanTranscript
import pke



def main():

	# Load pke model of choice

	# Give model a cleaned transcript
	transcript = ' '.join(readCleanTranscript("../preprocessing/clean_transcripts_june11.txt", 
										4))
	print(transcript)
	# Compare output to model used by us

	extractor = pke.unsupervised.YAKE()
	# extractor.load_document(input = "inputPKE.txt", language = 'nl')
	extractor.load_document(transcript, language = 'nl')


	print("PKE yake:")
	extractor.candidate_selection()
	extractor.candidate_weighting()
	keyphrases = extractor.get_n_best(n = 5)
	print(keyphrases)

	print("Our yake:")
	print(yake_keywords(transcript, n=5))






if __name__ == '__main__':
	main()