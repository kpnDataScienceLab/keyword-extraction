from tfidf import tfidf
from bm25 import bm25
from rake import rake
from git_pke import pke_multipartiterank, pke_positionrank, pke_singlerank, pke_textrank, pke_topicrank, pke_yake
from datasets.datasets import Dataset
from eval_metrics import get_results
import argparse
import csv
from tqdm import tqdm
from graphmodel import graphmodel
from datetime import datetime
import os

# skips useless warnings in the pke methods
import logging

logging.basicConfig(level=logging.CRITICAL)

global time_id


def save_results(name, dataset_name, ap_metrics, f1_metrics, k, match_type):
    """
    Save results or append them to an existing csv file
    """
    # if file doesn't exist, initialize it with the right columns
    if not os.path.isfile(f'evaluations/evaluations_{dataset_name}.csv'):
        with open(f'evaluations/evaluations_{dataset_name}.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                ["method"] + list(ap_metrics.keys()) + list(f1_metrics.keys()) + ['k', 'matching_type', 'time'])
            csv_writer.writerow(
                [name.lower()] + list(ap_metrics.values()) + list(f1_metrics.values()) + [k, match_type, time_id])
    else:
        with open(f'evaluations/evaluations_{dataset_name}.csv', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(
                [name.lower()] + list(ap_metrics.values()) + list(f1_metrics.values()) + [k, match_type, time_id])


def run_pipeline(name, train, test, arguments, k=10, dataset_name='DUC-2001', match_type='strict'):
    print()
    print(f'Evaluating {name.upper()} on {dataset_name}')
    print()

    # loading the dataset
    dataset = Dataset(dataset_name)

    # train whichever method we're using
    print('Training the model...')
    train(dataset.texts, arguments=arguments, lang='english')

    print('Running predictions...')
    predictions = []
    for idx, text in tqdm(enumerate(dataset.texts), ncols=80, smoothing=0.15, total=len(dataset)):
        try:
            predictions.append(test(text, arguments=arguments, k=k, lang='english'))
        except ValueError:
            tqdm.write(f"[WARNING] Skipping text {idx + 1} due to ValueError.")
            predictions.append([])

    print(f'Calculating scores...')
    results = get_results(dataset.labels, predictions, k=k, match_type=match_type)

    print(f"AP scores {name}:")
    for key in results['ap']:
        print(f"{key}:".rjust(15) + f"{results['ap'][key]:.3f}".rjust(7))

    print()
    print(f"F1 scores {name}:")
    for key in results['f1']:
        print(f"{key}:".rjust(15) + f"{results['f1'][key]:.3f}".rjust(7))

    save_results(name, dataset_name, results['ap'], results['f1'], k, match_type)


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
        "--k",
        type=int,
        help="Cutoff for the keyword extraction method and for the score calculations",
        default=10
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
        choices=['strict', 'levenshtein', 'spacy'],
        help="Matching function to use when evaluating keyword similarity",
        default='strict'
    )

    args = parser.parse_args()

    if args.dataset == 'all':
        datasets = ['500N-KPCrowd', 'DUC-2001', 'Inspec', 'SemEval-2010', 'NUS', 'WWW', 'KDD']
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        if args.mprank is not None:
            if len(args.mprank) < 3:
                args.mprank = ['1.1', '0.74', 'average']
                # print("MPRank: Alpha, threshold and method parameters not given.",
                #       "\nUsing default values: ", args.mprank)
            else:
                # print("MPRank. Using arguments: ", args.mprank)
                pass
            methods.append({'name': 'MultiPartiteRank',
                            'train': pke_multipartiterank.train,
                            'test': pke_multipartiterank.test,
                            'arguments': args.mprank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.positionrank is not None:
            methods.append({'name': 'PositionRank',
                            'train': pke_positionrank.train,
                            'test': pke_positionrank.test,
                            'arguments': args.positionrank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.singlerank is not None:
            methods.append({'name': 'SingleRank',
                            'train': pke_singlerank.train,
                            'test': pke_singlerank.test,
                            'arguments': args.singlerank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.textrank is not None:
            methods.append({'name': 'TextRank',
                            'train': pke_textrank.train,
                            'test': pke_textrank.test,
                            'arguments': args.textrank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.tfidf is not None:
            methods.append({'name': 'tfidf',
                            'train': tfidf.train,
                            'test': tfidf.test,
                            'arguments': args.tfidf,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.bm25 is not None:
            methods.append({'name': 'bm25',
                            'train': bm25.train,
                            'test': bm25.test,
                            'arguments': args.bm25,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.rake is not None:
            methods.append({'name': 'rake',
                            'train': rake.train,
                            'test': rake.test,
                            'arguments': args.rake,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.yake is not None:
            methods.append({'name': 'yake',
                            'train': pke_yake.train,
                            'test': pke_yake.test,
                            'arguments': args.yake,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.graphmodel is not None:
            methods.append({'name': 'graphmodel',
                            'train': graphmodel.train,
                            'test': graphmodel.test,
                            'arguments': args.graphmodel,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

        if args.topicrank is not None:
            methods.append({'name': 'TopicRank',
                            'train': pke_topicrank.train,
                            'test': pke_topicrank.test,
                            'arguments': args.topicrank,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

    try:
        for m in methods:
            run_pipeline(**m)
    except KeyboardInterrupt:
        print("\nTerminating...")
        quit()
