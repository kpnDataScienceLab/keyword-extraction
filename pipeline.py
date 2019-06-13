from tfidf import tfidf
from bm25 import bm25
from rake import rake
from git_pke import pke_multipartiterank, pke_positionrank, pke_singlerank, pke_textrank, pke_topicrank, pke_yake
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1
import argparse
import csv
from tqdm import tqdm
from graphmodel import graphmodel
import os


def save_results(name, dataset_name, ap_metrics, f1_metrics):
    """
    Save results or append them to an existing csv file
    """
    # if file doesn't exist, initialize it with the right columns
    if not os.path.isfile(f'evaluations/evaluations_{dataset_name}.csv'):
        with open(f'evaluations/evaluations_{dataset_name}.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Method"] + list(ap_metrics.keys()) + list(f1_metrics.keys()))
            csv_writer.writerow([name] + list(ap_metrics.values()) + list(f1_metrics.values()))
    else:
        with open(f'evaluations/evaluations_{dataset_name}.csv', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([name] + list(ap_metrics.values()) + list(f1_metrics.values()))


def run_pipeline(name, train, test, arguments, k=10, dataset_name='DUC-2001'):
    print()
    print(f'Evaluating {name}...')

    # loading the dataset
    dataset = Dataset(dataset_name)

    # train whichever method we're using
    train(dataset.texts, arguments=arguments, lang='english')

    predictions = []
    for text in tqdm(dataset.texts, ncols=80):
        predictions.append(test(text, arguments=arguments, k=k, lang='english'))
<<<<<<< HEAD
        # print(f'predictions: {predictions[-1]}')
        # print(f'targets: {dataset.labels[len(predictions)-1]}')
    print(f'calculating scores {name}...') 
    ap_metrics = mean_ap(dataset.labels, predictions, k=k)
    f1_metrics = mean_f1(dataset.labels, predictions, k=k)
=======

    ap_metrics = mean_ap(dataset.labels, predictions, k=k, loose=True)
    f1_metrics = mean_f1(dataset.labels, predictions, k=k, loose=True)
>>>>>>> b61ce238cfa15b1d39a0b5c4d67a67eb16644b6d

    print(f"AP scores {name}:")
    for key in ap_metrics:
        print(f"{key}:".rjust(15) + f"{ap_metrics[key]:.3f}".rjust(7))

    print()
    print(f"F1 scores {name}:")
    for key in f1_metrics:
        print(f"{key}:".rjust(15) + f"{f1_metrics[key]:.3f}".rjust(7))

    save_results(name, dataset_name, ap_metrics, f1_metrics)


if __name__ == "__main__":
    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mprank",
        help = "Use MultiPartiteRank",
        nargs = "*"
    )

    # --------------------------------- all flags and methods below are working

    parser.add_argument(
        "--positionrank",
        help = "Use PositionRank",
        nargs = "*"
    )

    parser.add_argument(
        "--singlerank",
        help = "Use SingleRank",
        nargs = "*"
    )

    parser.add_argument(
        "--textrank",
        help = "Use TextRank",
        nargs = "*",
    )

    parser.add_argument(
        "--topicrank",
        help = "Use TopicRank",
        nargs = '*',
    )

    parser.add_argument(
        "--yake",
        help="Use YAKE",
        nargs='*',
    )

    parser.add_argument(
        "--bm25",
        help="Use BM25",
        nargs='*',
    )

    parser.add_argument(
        "--tfidf",
        help="Use tf-idf",
        nargs='*',
    )

    parser.add_argument(
        "--rake",
        help="Use RAKE",
        nargs='*',
    )
    parser.add_argument(
        "--graphmodel",
        help="Use GRAPHMODEL",
        nargs='*',
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
        choices=['500N-KPCrowd', 'DUC-2001', 'Inspec'],
        help="Dataset to be used",
        default='DUC-2001'
    )

    args = parser.parse_args()

    if args.mprank is not None:
        methods.append({'name': 'MultiPartiteRank',
                        'train': pke_multipartiterank.train,
                        'test': pke_multipartiterank.test,
                        'arguments': args.mprank,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.positionrank is not None:
        methods.append({'name': 'PositionRank',
                        'train': pke_positionrank.train,
                        'test': pke_positionrank.test,
                        'arguments': args.positionrank,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.singlerank is not None:
        methods.append({'name': 'SingleRank',
                        'train': pke_singlerank.train,
                        'test': pke_singlerank.test,
                        'arguments': args.singlerank,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.textrank is not None:
        methods.append({'name': 'TextRank',
                        'train': pke_textrank.train,
                        'test': pke_textrank.test,
                        'arguments': args.textrank,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.tfidf is not None:
        methods.append({'name': 'tfidf',
                        'train': tfidf.train,
                        'test': tfidf.test,
                        'arguments': args.tfidf,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.bm25 is not None:
        methods.append({'name': 'bm25',
                        'train': bm25.train,
                        'test': bm25.test,
                        'arguments': args.bm25,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.rake is not None:
        methods.append({'name': 'rake',
                        'train': rake.train,
                        'test': rake.test,
                        'arguments': args.rake,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.yake is not None:
        methods.append({'name': 'yake',
                        'train': pke_yake.train,
                        'test': pke_yake.test,
                        'arguments': args.yake,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.graphmodel is not None:
        methods.append({'name': 'graphmodel',
                        'train': graphmodel.train,
                        'test': graphmodel.test,
                        'arguments': args.graphmodel,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )

    if args.topicrank is not None:
        methods.append({'name': 'TopicRank',
                        'train': pke_topicrank.train,
                        'test': pke_topicrank.test,
                        'arguments': args.topicrank,
                        'k': args.k,
                        'dataset_name': args.dataset}
                       )        

    try:
        for m in methods:
            run_pipeline(**m)
    except KeyboardInterrupt:
        print("Terminating...")
        quit()
