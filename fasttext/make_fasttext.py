import io
import pickle
import pandas as pd
import string


def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()

    return text


def get_data_vocabulary():
    fname = '../aligned_epg_transcriptions_npo1_npo2.csv'
    texts = pd.read_csv(fname)["text"]

    # clean and stem texts, and store all unique words
    vocab = set()
    for t in texts:
        clean_t = clean_text(t)


# TODO: fix this to read one line at a time
def get_embeddings():
    fname = 'cc.nl.300.vec'

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    data = {}

    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])


def main():
    # get a set containing all stemmed words in the dataset
    vocabulary = get_data_vocabulary()
    embeddings = get_embeddings()
    with open('fasttext_embeddings.pkl', 'wb') as handle:
        pickle.dump(embeddings, handle)


if __name__ == '__main__':
    main()
