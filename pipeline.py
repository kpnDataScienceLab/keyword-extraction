from tfidf import tfidf
import pandas as pd
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1, average_precision, f1
import argparse
import csv
from tqdm import tqdm


def train_method(name, train, test, arguments, n=10, datasetname='500N-KPCrowd'):
    print(f'Evaluating {name}...')
    dataset = Dataset(datasetname)
    train(dataset.texts, arguments=arguments, lang='english')

    predictions = []

    for (text, targets) in tqdm(dataset, ncols=100):
        predictions.append(test(text, arguments=arguments, n=n))

    ap_metrics = mean_ap(dataset.labels, predictions, k=n)
    f1_metrics = mean_f1(dataset.labels, predictions, k=n)

    print(f"AP scores {name}:")
    for key in ap_metrics:
        print(f"{key}:".rjust(12) + f"{ap_metrics[key]:.3f}".rjust(7))

    print()
    print(f"F1 scores {name}:")
    for key in f1_metrics:
        print(f"{key}:".rjust(12) + f"{f1_metrics[key]:.3f}".rjust(7))

    with open('evaluations.csv', mode='w+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([name] + list(ap_metrics.values()))


if __name__ == "__main__":
    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        help="train tfidf",
        nargs='*',
    )

    parser.add_argument(
        "--k",
        type=int,
        help="train tfidf",
        default=10
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=['500N-KPCrowd', 'DUC-2001', 'Inspec'],
        help="train tfidf",
        default='500N-KPCrowd'
    )

    args = parser.parse_args()

    if not args.tfidf is None:
        methods.append(('tfidf',
                        tfidf.train,
                        tfidf.test,
                        args.tfidf,
                        args.k,
                        args.dataset)
                       )

    for m in methods:
        train_method(*m)