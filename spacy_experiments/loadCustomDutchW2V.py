import spacy 
import zipfile 
import gensim
from gensim.models import Word2Vec

"""
Given a zip-file of dutch word2vec embeddings, 
load them into an emtpy spacy-nlp object and return it.
Uses Dutch as default language and takes a 
parameter to see which type of loading function is required.
"""

# Function arguments:
# embedding: either "nordic" or any other string, if "nordic" use the 
#	KeyedVectors class to load embedding into a gensim object, else 
#	use a Word2Vec class to load embedding.
# pathToZip: a "string" of the path to the zip folder containing 
# 	the word2vec embeddings
# nameOfFile: the "string" name of the file as it appears inside the 
#	zip-folder

def word2VecToSpacy(pathToZip, nameOfFile):

	# 0. Load default spacy dutch language module
	dutch_nlp = spacy.load('nl_core_news_sm')

	# 1. Retrieve stream from zipfile
	with zipfile.ZipFile(pathToZip, "r") as archive:
		stream = archive.open(nameOfFile)

	# 2. Read embedding into a gensim object
	print("Loading word2vec embedding.. May take a couple of minutes..")
	model = gensim.models.KeyedVectors.load_word2vec_format(stream,
			binary = False, unicode_errors = 'replace')

	# 3. Load the keys from the gensim object
	keys = []
	for idx in range(len(model.vocab)):
		keys.append(model.index2word[idx])

	# 4. Update spacy vocabulary vector embeddings
	dutch_nlp.vocab.vectors = spacy.vocab.Vectors(data = model.syn0,
													keys = keys)

	# 5. Test for success
	sent1 = dutch_nlp("De hond is blij.")
	sent2 = dutch_nlp("De hond is boos.")
	if sent1.similarity(sent2) > 0.9:
		print(sent1.text + ". success!")
	else:
		print(sent2.text + ". failure!")

	return dutch_nlp

