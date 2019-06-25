import numpy as np
from models.graphmodel.graphmodel import get_model

optim_dict = {}


def spacy_check(keyword, k_list, threshold=0.85):
    """
    Checks a keyword against a list of keywords, returning true if the keyword is at a Levenshtein distance
    of less or equal than the threshold compared to any word in the list.
    """
    for k in k_list:
        try:
            similarity = optim_dict[(keyword, k)]
        except KeyError:
            similarity = optim_dict[(keyword, k)] = get_model().nlp(keyword).similarity(get_model().nlp(k))
        if similarity > threshold:
            return True
    return False


def intersect(s1, s2):
    s1 = set(s1.split())
    s2 = set(split for sent in s2 for split in sent.split())

    return len(s1 & s2) > 0


def levenshtein(s1, s2):
    """
    Lehvenstein distance implementation found on
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    :param s1: First string
    :param s2: Second string
    :return: Levenshtein distance between the two strings
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_check(keyword, k_list, threshold=1):
    """
    Checks a keyword against a list of keywords, returning true if the keyword is at a Levenshtein distance
    of less or equal than the threshold compared to any word in the list.
    """
    for k in k_list:
        if levenshtein(keyword, k) <= threshold:
            return True
    return False


def true_positive_check(keyword, labels, predictions, match_type):
    """
    This function handles the main check for whether a found keyphrase is a true positive, by checking
    whether (1) it is relevant, and (2) it is not a repetition.
    :param keyword: The keyword that is being evaluated.
    :param labels: The list of reference keywords.
    :param predictions: The remaining keywords that have been predicted by the extraction method.
    :param match_type: The method used for determining whether two keywords are equivalent.
    :return: True or False depending on the metric.
    """
    if match_type == 'strict':
        return keyword in labels and keyword not in predictions

    elif match_type == 'intersect':
        return intersect(keyword, labels) and keyword not in predictions
        # return intersect(keyword, labels) and not intersect(keyword, predictions)

    elif match_type == 'levenshtein':
        return levenshtein_check(keyword, labels) and not levenshtein_check(keyword, predictions)

    elif match_type == 'spacy':
        return spacy_check(keyword, labels) and not spacy_check(keyword, predictions)


def f1(labels, predictions, k=10, match_type='strict', debug=False):
    """
    Returns the F1 score between the two provided lists.
    :param labels: List of ground truth labels, which order doesn't matter.
    :param predictions: List of predicted keywords. The order here matters.
    :param k: The cutoff value for computing the average precision. If k==0, all items are used.
    :param match_type: The type of matching function to use when evaluating keyword similarity.
    :param debug: Whether to print the labels and correct predictions for each text.
    :return: The F1 score for the current list.
    """

    # make all keywords lowercase
    labels = [l.lower() for l in labels]
    predictions = [p.lower() for p in predictions]

    if not labels or not predictions:
        return 0.0

    if k:
        predictions = predictions[0:k] if len(predictions) >= k else predictions

    tp = 0.

    if debug:
        print('#' * 100)
        print(f"\n\nLabels:\n\n{labels}")
        print(f"\n\nCorrect predictions:\n\n[", end='')

    for i, p in enumerate(predictions):
        # check for relevance and avoid repetitions
        if true_positive_check(p, labels, predictions[:i], match_type):

            if debug:
                print(f"'{p}', ", end='')

            tp += 1.0

    if debug:
        print("\b\b]\n\n")

    tp_fp = len(predictions)
    tp_fn = min(len(labels), k) if k else len(labels)

    if tp == 0.:
        return 0.

    precision = tp / tp_fp
    recall = tp / tp_fn

    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def get_results(labels_list, predictions_list, k=10, match_type='strict', debug=False):
    """
    Returns the mean F1 score for a series of ranking attempts.
    :param labels_list: A list of lists, where each list contains the ground truth keywords for a text.
    :param predictions_list: A list of lists, where each list contains the predicted keywords for a text.
    :param k: The cutoff value for computing the average precision. With k==0, all items are used.
    :param match_type: The type of matching function to use when evaluating keyword similarity.
    :param debug: Whether to print the labels and correct predictions for each text.
    :return: A dictionary with 5 statistics from the list of f1 scores
    """
    f1_scores = []

    for labels, predictions in zip(labels_list, predictions_list):
        f1_scores.append(f1(labels, predictions, k, match_type, debug))

    results = {'f1_mean': np.mean(f1_scores),
               'f1_std': np.std(f1_scores),
               'f1_min': np.min(f1_scores),
               'f1_max': np.max(f1_scores),
               'f1_median': np.median(f1_scores)}

    return results

