from datetime import datetime
from rake_nltk import Rake
from tqdm import tqdm
import pickle as pkl
import argparse

import utils


def get_keywords(rake, text):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()


def extract_keywords(flags):
    # load texts from the sample dataset
    texts = utils.get_texts(flags.data_path)

    # identification for the current run
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_rake_" + flags.stopwords + '_' + flags.max_length)

    # The ranking_metric parameter may be:
    #  (1) [Metric.DEGREE_TO_FREQUENCY_RATIO] Ratio of degree of word to its frequency (default)
    #  (2) [Metric.WORD_DEGREE] Degree of word only
    #  (3) [Metric.WORD_FREQUENCY] Frequency of word only

    # select stopword list
    if flags.stopwords == 'nltk':
        stopwords = None
    else:
        stopwords = utils.get_stopwords(flags.stopwords)

    rake = Rake(
        language="dutch",
        stopwords=stopwords,
        max_length=flags.max_length
    )

    extracted = []
    print("Extracting keywords...")
    for text in tqdm(texts, ncols=80):
        extracted.append({"text": text, "keywords": get_keywords(rake, text)})

    with open("processed/" + run_id + ".pkl", "wb") as handle:
        pkl.dump(extracted, handle)


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
        default=99,
        help="Maximum length of each keyphrase"
    )
    parser.add_argument(
        "--stopwords",
        type=str,
        default="nltk",
        choices=['nltk', 'medium', 'large'],
        help="List of stopwords to be used"
    )

    flags = parser.parse_args()
    extract_keywords(flags)


main()
