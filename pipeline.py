from tfidf import tfidf
from bm25 import bm25
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1
import argparse
import csv
from tqdm import tqdm


def run_pipeline(name, train, test, arguments, k=10, dataset_name='DUC-2001'):
    print()
    print(f'Evaluating {name}...')

    # loading the dataset
    dataset = Dataset(dataset_name)

    # train whichever method we're using
    train(dataset.texts, arguments=arguments, lang='english')

    predictions = []
    for text in tqdm(dataset.texts, ncols=100):
        predictions.append(test(text, arguments=arguments, n=k, lang='english'))

    ap_metrics = mean_ap(dataset.labels, predictions, k=k)
    f1_metrics = mean_f1(dataset.labels, predictions, k=k)

    print(f"AP scores {name}:")
    for key in ap_metrics:
        print(f"{key}:".rjust(12) + f"{ap_metrics[key]:.3f}".rjust(7))

    print()
    print(f"F1 scores {name}:")
    for key in f1_metrics:
        print(f"{key}:".rjust(12) + f"{f1_metrics[key]:.3f}".rjust(7))

    with open(f'evaluations_{name}_{dataset_name}.csv', mode='w+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([name] + list(ap_metrics.values()))


if __name__ == "__main__":
    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfidf",
        help="Use tf-idf",
        nargs='*',
    )

    parser.add_argument(
        "--bm25",
        help="Use BM25",
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

    for m in methods:
        run_pipeline(**m)
