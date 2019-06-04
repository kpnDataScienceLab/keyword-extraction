from rake.rake import rake
from our_yake.run_yake import yake_keywords
from tfidf.tfidf import tfidf
import pandas as pd

file_name = 'aligned_epg_transcriptions_npo1_npo2.csv'
data = pd.read_csv(file_name)
texts = data['text']

text = texts[0]

n = 5
tfidf_words = tfidf(text, n=n)
rake_words = rake(text, n=n)
yake_words = yake_keywords(text, n=n)

print("index. tf-idf | rake | yake")
for idx in range(n):
    print(f"{idx + 1}) {tfidf_words[idx]} | {rake_words[idx]} | {yake_words[idx]}")
