from rake.rake import rake
from our_yake.run_yake import yake_keywords
from tfidf.tfidf import tfidf
from bm25.bm25 import bm25
import pandas as pd

file_name = 'aligned_epg_transcriptions_npo1_npo2.csv'
data = pd.read_csv(file_name)
texts = data['text']

n = 8

for i, text in enumerate(texts[0:5]):

    tfidf_words = tfidf(text, n=n)
    bm25_words = bm25(text, n=n)
    rake_words = rake(text, n=n)
    yake_words = yake_keywords(text, n=n)

    print()
    print(f"Document {i + 1}:")
    for idx in range(n):
        if idx >= len(tfidf_words):
            tfidf_words.append("---")
        if idx >= len(rake_words):
            rake_words.append("---")
        if idx >= len(yake_words):
            yake_words.append("---")
        print(f"{idx + 1}. [TF-IDF]: {tfidf_words[idx]} \t\t[BM25]: {bm25_words[idx]} "
              f"\t\t[RAKE]: {rake_words[idx]} \t\t[YAKE]: {yake_words[idx]}")

