from tfidf import tfidf
from git_pke import pke_multipartiterank, pke_topicrank
from datasets.datasets import Dataset
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import argparse
from eval_metrics import intersect
import traceback
import csv
import numpy as np

# skips useless warnings in the pke methods
import logging

logging.basicConfig(level=logging.CRITICAL)


def save_output(texts, keywords):
    with open(f'{time_id}_keywords_ensemble.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["texts", "labels"])
        for text, keyword in zip(texts, keywords):
            csv_writer.writerow([text, keyword])


def get_with_scores(keyword_list):
    total = len(keyword_list)
    scores = np.linspace(1, 0, total) ** 2
    keywords_with_scores = []
    for k, s in zip(keyword_list, scores):
        keywords_with_scores.append((k, s))
    return keywords_with_scores


def rank_match_keywords(temp_keywords, softening):
    # go through every found keyword, rank them based on occurrence amount
    scored_keywords = []
    idx = 0
    while idx < len(temp_keywords):

        # dict containing keywords registered as the same one, with their collective count
        current_keyword = {'words': [temp_keywords[idx][0]], 'score': temp_keywords[idx][1]}

        remaining_idx = idx + 1
        while remaining_idx < len(temp_keywords):

            # if the two keywords match (using the intersect method), add 1 to the keyword's score
            if intersect(temp_keywords[idx][0], [temp_keywords[remaining_idx][0]]):

                if temp_keywords[remaining_idx][0] not in current_keyword['words']:
                    current_keyword['words'].append(temp_keywords[remaining_idx][0])

                current_keyword['score'] = (current_keyword['score'] + temp_keywords[remaining_idx][1]) * softening
                del temp_keywords[remaining_idx]

            remaining_idx += 1

        scored_keywords.append(current_keyword)
        idx += 1

    # store the keywords ranked according to their occurrence count
    ranked_keywords = [keyword['words'] for keyword in
                       sorted(scored_keywords, reverse=True, key=lambda x: x['score'])]
    return ranked_keywords


def get_top_keywords(temp_keywords):
    sorted_keywords = [keyword[0] for keyword in sorted(temp_keywords, reverse=True, key=lambda x: x[1])]

    idx = 0
    while idx < len(sorted_keywords):

        current_keyword = sorted_keywords[idx]

        remaining_idx = idx + 1
        while remaining_idx < len(sorted_keywords):

            # if the two keywords match (using the intersect method) remove the second word by score
            if intersect(current_keyword, [sorted_keywords[remaining_idx]]):
                del sorted_keywords[remaining_idx]

            remaining_idx += 1

        idx += 1
    return sorted_keywords


def main(args):
    # parameters
    softening = 0.6

    texts = pd.read_csv('datasets/transcriptions_fixed.csv')['fixed_text'].tolist()
    texts = [text for text in texts if type(text) is str]

    # train TF-IDF
    tfidf.train(texts, arguments=None, lang='dutch')

    texts = texts[0:10]  # fixme

    keywords = []
    for text in tqdm(texts, ncols=80):
        temp_keywords = get_with_scores(tfidf.test(text, arguments=None, k=args.k, lang='dutch'))
        try:
            temp_keywords.extend(
                get_with_scores(
                    pke_multipartiterank.test(text, arguments=['1.1', '0.74', 'average'], k=args.k, lang='dutch')) +
                get_with_scores(pke_topicrank.test(text, arguments=None, k=args.k, lang='dutch')))
        except ValueError:
            tqdm.write(traceback.format_exc())

        ranked_keywords = get_top_keywords(temp_keywords)
        # ranked_keywords = rank_match_keywords(temp_keywords, softening) fixme

        keywords.append(ranked_keywords[0:args.n] if len(ranked_keywords) >= args.n else ranked_keywords)

    save_output(texts, keywords)


if __name__ == '__main__':
    # initialize the time id to identify the current run
    global time_id
    time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        help="Number of keywords each of the three models will return",
        type=int,
        default=30
    )

    parser.add_argument(
        "--n",
        help="Number of keywords returned for each text by the ensemble method",
        type=int,
        default=10
    )

    args = parser.parse_args()

    main(args)
