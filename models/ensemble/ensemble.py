from models.tfidf import tfidf
from models.git_pke import pke_topicrank
from models.git_pke import pke_multipartiterank
from tqdm import tqdm
from utils.eval_metrics import intersect
import traceback
import numpy as np

# skips useless warnings in the pke methods
import logging

logging.basicConfig(level=logging.CRITICAL)


def get_with_scores(keyword_list):
    total = len(keyword_list)
    scores = np.linspace(1, 0, total) ** 2
    keywords_with_scores = []
    for k, s in zip(keyword_list, scores):
        keywords_with_scores.append((k, s))
    return keywords_with_scores


def rank_matched_keywords(temp_keywords, softening):
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


# currently unused
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


def train(dataset, arguments=None, lang='dutch'):
    # train TF-IDF
    tfidf.train(dataset, arguments, lang)


def test(text, arguments=['40'], k=20, lang='dutch'):
    # affects the summing of scores for reoccurring keyphrases
    softening = 0.6

    # how many keywords each method inidividually should return
    if not arguments:
        n = 40
    else:
        n = int(arguments([0]))

    # get tfidf keyphrases along with scores proportional to their rank in the list (from 1 to 0)
    temp_keywords = get_with_scores(tfidf.test(text, arguments=[], k=n, lang='dutch'))

    # get multipartiterank and topcrank keyphrases, again with their score
    try:
        temp_keywords.extend(
            get_with_scores(
                pke_multipartiterank.test(text, arguments=['1.1', '0.74', 'average'], k=n, lang='dutch')) +
            get_with_scores(pke_topicrank.test(text, arguments=[], k=n, lang='dutch')))
    except ValueError:
        tqdm.write(traceback.format_exc())

    # rank keywords by score, and by combining the scores of reoccurring keywords
    ranked_keywords = rank_matched_keywords(temp_keywords, softening)

    return ranked_keywords[0:k] if len(ranked_keywords) >= k else ranked_keywords
