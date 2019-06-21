from models.tfidf import tfidf
from models.bm25 import bm25
from models.rake import rake
from models.git_pke import pke_textrank, pke_topicrank
from models.git_pke import pke_singlerank, pke_multipartiterank, pke_positionrank, pke_yake
from models.ensemble import ensemble
from datasets.datasets import Dataset
from eval_metrics import get_results
import argparse
import csv
from tqdm import tqdm
from models.graphmodel import graphmodel
from datetime import datetime
import os
import traceback

# skips useless warnings in the pke methods
import logging

logging.basicConfig(level=logging.CRITICAL)

global time_id


def save_results(name, dataset_name, f1_metrics, k, match_type):
    """
    Save results or append them to an existing csv file
    """

    # create the destination folder if it doesn't exist
    if not os.path.exists('evaluations'):
        os.mkdir('evaluations')
        print(f"Created evaluations folder")

    # if file doesn't exist, initialize it with the right columns
    if not os.path.isfile(f'evaluations/evaluations_{dataset_name}.csv'):
        with open(f'evaluations/evaluations_{dataset_name}.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ["method"] + list(f1_metrics.keys()) + ['k', 'matching_type', 'time'])
            csv_writer.writerow(
                [name.lower()] + list(f1_metrics.values()) + [k, match_type, time_id])

    # the file already exists, so just append the results
    else:
        with open(f'evaluations/evaluations_{dataset_name}.csv', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [name.lower()] + list(f1_metrics.values()) + [k, match_type, time_id])

    print("Saved results")


def run_pipeline(name, train_function, test_function, arguments, k=10, dataset_name='DUC-2001', match_type='strict'):
    print(f'\nEvaluating {name.upper()} on {dataset_name}\n')

    # loading the dataset
    dataset = Dataset(dataset_name)

    # train whichever method we're using
    print('Training the model...')
    train_function(dataset.texts, arguments=arguments, lang='english')

    print('Running predictions...')
    predictions = []
    for idx, (text, label) in tqdm(enumerate(zip(dataset.texts, dataset.labels)), ncols=80, smoothing=0.15,
                                   total=len(dataset)):
        try:
            predictions.append(test_function(text, arguments=arguments, k=k, lang='english'))
        except ValueError:
            tqdm.write(traceback.format_exc())
            predictions.append([])

    print(f'Calculating scores...')
    results = get_results(dataset.labels, predictions, k=k, match_type=match_type, debug=(not __debug__))

    print(f"F1 scores for {name.upper()}:")
    for key in results:
        print(f"{key}:".rjust(15) + f"{results[key]:.3f}".rjust(7))

    save_results(name, dataset_name, results, k, match_type)


if __name__ == "__main__":

    # initialize the time id to identify the current run
    global time_id
    time_id = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mprank",
        help="Use MultiPartiteRank",
        nargs="*",
    )

    parser.add_argument(
        "--positionrank",
        help="Use PositionRank",
        nargs="*"
    )

    parser.add_argument(
        "--singlerank",
        help="Use SingleRank",
        nargs="*"
    )

    parser.add_argument(
        "--textrank",
        help="Use TextRank",
        nargs="*"
    )

    parser.add_argument(
        "--topicrank",
        help="Use TopicRank",
        nargs='*'
    )

    parser.add_argument(
        "--yake",
        help="Use YAKE",
        nargs='*'
    )

    parser.add_argument(
        "--bm25",
        help="Use BM25",
        nargs='*'
    )

    parser.add_argument(
        "--tfidf",
        help="Use tf-idf",
        nargs='*',
    )

    parser.add_argument(
        "--rake",
        help="Use RAKE",
        nargs='*'
    )

    parser.add_argument(
        "--graphmodel",
        help="Use GRAPHMODEL",
        nargs='*'
    )

    parser.add_argument(
        "--ensemble",
        help="Use the ensemble model",
        nargs='*'
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Cutoff for the keyword extraction method and for the score calculations",
        default=20
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=['500N-KPCrowd', 'DUC-2001', 'Inspec', 'SemEval-2010', 'NUS', 'WWW', 'KDD', 'all'],
        help="Dataset to be used",
        default='DUC-2001'
    )

    parser.add_argument(
        "--matchtype",
        type=str,
        choices=['strict', 'levenshtein', 'spacy', 'intersect'],
        help="Matching function to use when evaluating keyword similarity",
        default='strict'
    )

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['500N-KPCrowd', 'DUC-2001', 'Inspec', 'NUS', 'WWW', 'KDD']
    else:
        datasets = [args.dataset]

    for dataset in datasets:

        if args.mprank is not None:
            if len(args.mprank) < 3:
                args.mprank = ['1.1', '0.74', 'average']

            methods.append({'name': 'MultiPartiteRank',
                            'train_function': pke_multipartiterank.train,
                            'test_function': pke_multipartiterank.test,
                            'arguments': args.mprank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.positionrank is not None:
            methods.append({'name': 'PositionRank',
                            'train_function': pke_positionrank.train,
                            'test_function': pke_positionrank.test,
                            'arguments': args.positionrank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.singlerank is not None:
            methods.append({'name': 'SingleRank',
                            'train_function': pke_singlerank.train,
                            'test_function': pke_singlerank.test,
                            'arguments': args.singlerank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.textrank is not None:
            methods.append({'name': 'TextRank',
                            'train_function': pke_textrank.train,
                            'test_function': pke_textrank.test,
                            'arguments': args.textrank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.tfidf is not None:
            methods.append({'name': 'tfidf',
                            'train_function': tfidf.train,
                            'test_function': tfidf.test,
                            'arguments': args.tfidf,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.bm25 is not None:
            methods.append({'name': 'bm25',
                            'train_function': bm25.train,
                            'test_function': bm25.test,
                            'arguments': args.bm25,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.rake is not None:
            methods.append({'name': 'rake',
                            'train_function': rake.train,
                            'test_function': rake.test,
                            'arguments': args.rake,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.yake is not None:
            methods.append({'name': 'yake',
                            'train_function': pke_yake.train,
                            'test_function': pke_yake.test,
                            'arguments': args.yake,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.graphmodel is not None:
            methods.append({'name': 'graphmodel',
                            'train_function': graphmodel.train,
                            'test_function': graphmodel.test,
                            'arguments': args.graphmodel,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.topicrank is not None:
            methods.append({'name': 'TopicRank',
                            'train_function': pke_topicrank.train,
                            'test_function': pke_topicrank.test,
                            'arguments': args.topicrank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.ensemble is not None:
            methods.append({'name': 'Ensemble',
                            'train_function': ensemble.train,
                            'test_function': ensemble.test,
                            'arguments': args.ensemble,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

    try:
        for m in methods:
            run_pipeline(**m)
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Terminating...")
        quit()
