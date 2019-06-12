from tfidf import tfidf
import pandas as pd
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1, average_precision, f1
import argparse
from tqdm import tqdm


def train_method(name, train, test, arguments, dataset='500N-KPCrowd'):
    dataset = Dataset(dataset)
    texts_labels = dataset.get_texts()
    train([t[0] for t in texts_labels], arguments=arguments, lang='english')

    predictions = []

    for (text, targets) in tqdm(dataset, ncols=100):
        predictions.append(test(text, arguments=arguments, n=len(targets)))

    ap_metrics = mean_ap([t[1] for t in texts_labels], predictions)
    f1_metrics = mean_f1([t[1] for t in texts_labels], predictions)

    print("AP scores:")
    for key in ap_metrics:
        print(f"\t{key}: {ap_metrics[key]}")

    print()
    print("F1 scores:")
    for key in f1_metrics:
        print(f"\t{key}: {f1_metrics[key]}")


if __name__ == "__main__":

    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tfidf",
        action="store",
        help="train tfidf",
        nargs='*',
    )

    args = parser.parse_args()

    if not args.tfidf is None:
        methods.append(('tfidf', tfidf.train, tfidf.test, args.tfidf))

    for m in methods:
        train_method(*m)
