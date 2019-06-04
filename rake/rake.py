from datetime import datetime
from rake_nltk import Rake
from tqdm import tqdm
import pickle as pkl
import argparse
import string

import rake.utils as utils


def get_keywords(rake, text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()


def save_keywords(extracted, run_id):
    with open("processed/" + run_id + ".pkl", "wb") as handle:
        pkl.dump(extracted, handle)


def rake(text,
         n=20,
         max_length=3,
         stopwords='nltk',
         punctuation=False):

    # select stopword list
    if stopwords == 'nltk':
        stopwords = None
    else:
        stopwords = utils.get_stopwords(stopwords)

    r = Rake(
        language="dutch",
        stopwords=stopwords,
        max_length=max_length
    )

    if not punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    keywords = get_keywords(r, text)
    return keywords[0:n] if len(keywords) >= n else keywords


def process_dataset(data_path='../aligned_epg_transcriptions_npo1_npo2.csv',
                    max_length=3,
                    stopwords='nltk',
                    punctuation=False):
    # load texts from the sample dataset
    texts = utils.get_texts(data_path)

    # identification for the current run
    run_id = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_rake_{stopwords}_{max_length}")

    # The ranking_metric parameter may be:
    #  (1) [Metric.DEGREE_TO_FREQUENCY_RATIO] Ratio of degree of word to its frequency (default)
    #  (2) [Metric.WORD_DEGREE] Degree of word only
    #  (3) [Metric.WORD_FREQUENCY] Frequency of word only

    # select stopword list
    if stopwords == 'nltk':
        stopwords = None
    else:
        stopwords = utils.get_stopwords(stopwords)

    rake = Rake(
        language="dutch",
        stopwords=stopwords,
        max_length=max_length
    )

    extracted = []
    print("Extracting keywords...")
    for text in tqdm(texts, ncols=80):
        if not punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        extracted.append({"text": text, "keywords": get_keywords(rake, text)})

    save_keywords(extracted, run_id)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="../aligned_epg_transcriptions_npo1_npo2.csv",
        help="Path to the csv file containing the data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=3,
        help="Maximum length of each keyphrase"
    )
    parser.add_argument(
        "--stopwords",
        type=str,
        default="nltk",
        choices=['nltk', 'medium', 'large'],
        help="List of stopwords to be used"
    )
    parser.add_argument(
        "--punctuation",
        action="store_true",
        help="Don't manually remove punctuation from the texts"
    )

    flags = parser.parse_args()
    process_dataset(flags.data_path, flags.max_length, flags.stopwords, flags.punctuation)


if __name__ == '__main__':
    main()
