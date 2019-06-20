import io
import pickle
import pandas as pd
import string
import re
from tqdm import tqdm


def clean_text(text):
    # remove punctuation
    punctuation = string.punctuation.replace('-', '')
    text = text.translate(str.maketrans(punctuation, ' ' * len(punctuation)))

    # remove numbers
    # numbers = '1234567890'
    # text = text.translate(str.maketrans(numbers, ' ' * len(numbers)))

    # remove leading or trailing dashes
    text = re.sub(r'- | -', ' ', text)

    # remove duplicate spaces
    text = re.sub(r' +', ' ', text)

    # make text lowercase
    text = text.lower()
    return text


def get_data_vocabulary():
    fname = '../datasets/transcriptions_fixed.csv'
    texts = pd.read_csv(fname)["fixed_text"]

    # clean and stem texts, and store all unique words
    vocab = set()
    for t in texts:
        clean_t = clean_text(t)
        vocab.update(clean_t.split(' '))

    return vocab


def get_embeddings(vocab):
    fname = 'embeddings/cc.nl.300.vec'

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    embeddings = {}

    for line in tqdm(fin, ncols=120, total=2000000):
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocab:
            embeddings[tokens[0]] = [float(t) for t in tokens[1:]]

    return embeddings


def main():
    # get a set containing all stemmed words in the dataset
    vocabulary = get_data_vocabulary()
    embeddings = get_embeddings(vocabulary)

    with open('fasttext_embeddings.pkl', 'wb') as handle:
        pickle.dump(embeddings, handle)


if __name__ == '__main__':
    main()
