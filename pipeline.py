from tfidf.tfidf import train,test
import pandas as pd

file_name = 'transcriptions.csv'
data = pd.read_csv(file_name)
texts = data['text']

train(texts)
print(test(texts[0]))