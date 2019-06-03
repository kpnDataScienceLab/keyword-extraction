import pandas as pd
import nltk

# tokenize lowercase text
def tokenizer(sentence):
    return [word.lower() for word in nltk.word_tokenize(sentence)]

# load csv file
file_name = 'aligned_epg_transcriptions_npo1_npo2.csv'
data = pd.read_csv(file_name)

# get list of texts
texts = data['text']