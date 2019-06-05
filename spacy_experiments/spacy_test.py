import pandas as pd
import nltk
import spacy 
import zipfile 
import gensim

def csvToTranscripts(filename = 'aligned_epg_transcriptions_npo1_npo2.csv'):

	data = pd.read_csv(filename)

	# get list of texts
	texts = data['text']
	channels = data['channel']

	return texts, channels

transcripts, _ = csvToTranscripts("../../aligned_epg_transcriptions_npo1_npo2.csv")

# Load model from the nordic language processing laboratory 
# (2 million words 2017!)
zip_file = "../../Word2VecModels_Dutch/Dutch_Word2Vec.zip"
with zipfile.ZipFile(zip_file, "r") as archive:
	stream = archive.open("Dutch_Word2Vec/model.txt")

# 1. How to load a model from the Nordic Language Processing Laboratory
model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=False, unicode_errors='replace')

print(model)
print(len(model.vocab))

# 2. Create a blank spacy object
nlp_nordicDutch = spacy.blank('nl')

# 3. Load the keys from the gensim object
keys = []
for idx in range(len(model.vocab)):
	keys.append(model.index2word[idx])

# 4. make spacy object

nlp_nordicDutch.vocab.vectors = spacy.vocab.Vectors(data=model.syn0,
	keys=keys)

doc_test = nlp_nordicDutch("De hond is blij")
doc_test2 = nlp_nordicDutch("De hond is boos")
print(doc_test.similarity(doc_test2))




# for t1 in range(7):
# 	for t2 in range(7):
# 		doc1 = nlp_dutch(transcripts[t1])
# 		doc2 = nlp_dutch(transcripts[t2])
# 		print("Similarity |", t1, '|', t2, " = ", doc1.similarity(doc2))
