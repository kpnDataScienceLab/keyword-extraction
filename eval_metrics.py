import numpy as np
import spacy
from graphmodel.graphmodel import train as graph_model_config
from tqdm import tqdm


def spacy_evaluate(prediction,labels):
    graph_model_config('','','english')
    from graphmodel.graphmodel import _model
    for l in labels:
        sim = _model.nlp(prediction).similarity(_model.nlp(l))
        if  sim > 0.85:
            print(f'match-> {prediction} : {l} @ {sim}')
            return True
    return False

def f1_spacy(labels, predictions, k=10):
    """
    Returns the F1 score between the two provided lists.
    :param labels: List of ground truth labels, which order doesn't matter.
    :param predictions: List of predicted keywords. The order here matters.
    :param k: The cutoff value for computing the average precision. If k==0, all items are used.
    :return: The F1 score for the current list.
    """
    if not labels or not predictions:
        return 0.0

    if k:
        predictions = predictions[0:k] if len(predictions) >= k else predictions

    tp = 0.

    for i, p in enumerate(predictions):
        # check for relevance and avoid repetitions
        if spacy_evaluate(p,labels):
            tp += 1.0

    tp_fp = min(len(labels), k) if k else len(labels)
    tp_fn = len(predictions)

    if tp == 0.:
        return 0.

    precision = tp / tp_fp
    recall = tp / tp_fn

    return (2 * precision * recall) / (precision + recall)

def ap_spacy(labels, predictions, k=10):
    if not labels or not predictions:
        return 0.0

    if k:
        predictions = predictions[0:k] if len(predictions) >= k else predictions

    score = 0.
    tp = 0.

    for i, p in enumerate(predictions):
        # check for relevance and avoid repetitions
        if spacy_evaluate(p,labels):
            tp += 1.0
            score += tp / (i + 1.0)

    if tp == 0.:
        return 0.

    tp_fp = min(len(labels), k) if k else len(labels)
    return score / tp_fp



def average_precision(labels, predictions, k=10):
    """
    Returns the average precision score between the two provided lists.
    :param labels: List of ground truth labels, which order doesn't matter.
    :param predictions: List of predicted keywords. The order here matters.
    :param k: The cutoff value for computing the average precision. If k==0, all items are used.
    :return: The average precision score for the current list.
    """
    if not labels or not predictions:
        return 0.0

    if k:
        predictions = predictions[0:k] if len(predictions) >= k else predictions

    score = 0.
    tp = 0.

    for i, p in enumerate(predictions):
        # check for relevance and avoid repetitions
        if p in labels and p not in predictions[:i]:
            tp += 1.0
            score += tp / (i + 1.0)

    if tp == 0.:
        return 0.

    tp_fp = min(len(labels), k) if k else len(labels)
    return score / tp_fp


def mean_ap(labels_list, predictions_list, k=10,spacy_=True):
    """
    Returns the mean average precision for a series of ranking attempts.
    :param labels_list: A list of lists, where each list contains the ground truth keywords for a text.
    :param predictions_list: A list of lists, where each list contains the predicted keywords for a text.
    :param k: The cutoff value for computing the average precision. With k==0, all items are used.
    :return: A dictionary with 5 statistics from the list of average precision scores
    """
    ap_scores = []
    if not spacy_:
        for labels, predictions in tqdm(zip(labels_list, predictions_list)):
            ap_scores.append(average_precision(labels, predictions, k))
    else:
        for labels, predictions in tqdm(zip(labels_list, predictions_list)):
            ap_scores.append(ap_spacy(labels, predictions, k))

    results = {'ap_mean': np.mean(ap_scores),
               'ap_std': np.std(ap_scores),
               'ap_min': np.min(ap_scores),
               'ap_max': np.max(ap_scores),
               'ap_median': np.median(ap_scores)}

    return results


def f1(labels, predictions, k=10):
    """
    Returns the F1 score between the two provided lists.
    :param labels: List of ground truth labels, which order doesn't matter.
    :param predictions: List of predicted keywords. The order here matters.
    :param k: The cutoff value for computing the average precision. If k==0, all items are used.
    :return: The F1 score for the current list.
    """
    if not labels or not predictions:
        return 0.0

    if k:
        predictions = predictions[0:k] if len(predictions) >= k else predictions

    tp = 0.

    for i, p in enumerate(predictions):
        # check for relevance and avoid repetitions
        if p in labels and p not in predictions[:i]:
            tp += 1.0

    tp_fp = min(len(labels), k) if k else len(labels)
    tp_fn = len(predictions)

    if tp == 0.:
        return 0.

    precision = tp / tp_fp
    recall = tp / tp_fn

    return (2 * precision * recall) / (precision + recall)


def mean_f1(labels_list, predictions_list, k=10,spacy_=True):
    """
    Returns the mean F1 score for a series of ranking attempts.
    :param labels_list: A list of lists, where each list contains the ground truth keywords for a text.
    :param predictions_list: A list of lists, where each list contains the predicted keywords for a text.
    :param k: The cutoff value for computing the average precision. With k==0, all items are used.
    :return: A dictionary with 5 statistics from the list of F1 scores
    """
    f1_scores = []
    if not spacy_:
        for labels, predictions in tqdm(zip(labels_list, predictions_list)):
            f1_scores.append(f1(labels, predictions, k))
    else:
        for labels, predictions in tqdm(zip(labels_list, predictions_list)):
            f1_scores.append(f1_spacy(labels, predictions, k))

    results = {'f1_mean': np.mean(f1_scores),
               'f1_std': np.std(f1_scores),
               'f1_min': np.min(f1_scores),
               'f1_max': np.max(f1_scores),
               'f1_median': np.median(f1_scores)}

    return results
