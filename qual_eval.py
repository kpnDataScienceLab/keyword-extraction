from rake.rake import rake
from our_yake.run_yake import yake_keywords
from tfidf.tfidf import tfidf
from bm25.bm25 import bm25
import pandas as pd


def main():
    file_name = 'aligned_epg_transcriptions_npo1_npo2.csv'
    data = pd.read_csv(file_name)
    texts = data['text']

    print()
    print("   " + "[TFIDF]".ljust(45) + "[BM25]".ljust(45) + "[RAKE]".ljust(45) + "[YAKE]")

    for i, text in enumerate(texts[0:5]):

        tfidf_words = tfidf(text, n=5)
        bm25_words = bm25(text, n=5)
        rake_words = rake(text, n=5)
        yake_words = yake_keywords(text, n=5)

        print()
        print(f"Document {i + 1}:")
        print()
        for idx in range(5):
            if idx >= len(tfidf_words):
                tfidf_words.append("---")
            if idx >= len(rake_words):
                rake_words.append("---")
            if idx >= len(yake_words):
                yake_words.append("---")
            if idx >= len(bm25_words):
                bm25_words.append("---")

            print(f"{idx + 1}. " + tfidf_words[idx].ljust(45) + bm25_words[idx].ljust(45)
                  + rake_words[idx].ljust(45) + yake_words[idx])


main()
