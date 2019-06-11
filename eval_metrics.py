import numpy as np


def average_precision(labels, predictions, k=10):
    """
    Returns the average precision score between the two provided lists.
    :param labels: List of ground truth labels, which order doesn't matter.
    :param predictions: List of predicted keywords. The order here matters.
    :param k: The cutoff value for computing the average precision. If k==0, all items are used.
    :return: The average precision score for the current list.
    """
    if not labels:
        return 0.0

    if k:
        predictions = predictions[0:k] if len(predictions) >= k else predictions

    score = 0.
    n_rel = 0.

    for i, p in enumerate(predictions):
        # check for relevance and avoid repetitions
        if p in labels and p not in predictions[:i]:
            n_rel += 1.0
            score += n_rel / (i + 1.0)

    total_n = min(len(labels), k) if k else len(labels)
    return score / total_n


def map(labels_list, predictions_list, k=10):
    """
    Returns the mean average precision for a series of ranking attempts.
    :param labels_list: A list of lists, where each list contains the ground truth keywords for a text.
    :param predictions_list: A list of lists, where each list contains the predicted keywords for a text.
    :param k: The cutoff value for computing the average precision. With k==0, all items are used.
    :return: The mean average precision score for the whole set.
    """
    ap = []

    for labels, predictions in zip(labels_list, predictions_list):
        ap.append(average_precision(labels, predictions, k))

    return np.mean(ap)