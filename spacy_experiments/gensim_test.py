import zipfile
import gensim

zip_file = "../Dutch_Word2Vec.zip"
with zipfile.ZipFile(zip_file, "r") as archive:
	stream = archive.open("Dutch_Word2Vec/model.txt")

print(stream)

# How to load a model from the Nordic Language Processing Laboratory
model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=False, unicode_errors='replace')

katvec = model['kat']
print(katvec)
result = model.most_similar('kat')
print(result)

# How to load a model from the Dutch word2vec 
# models from the dutchembeddings-repo
