from tfidf import tfidf
import pandas as pd
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1, average_precision, f1
import argparse
import csv
from tqdm import tqdm


def train_method(name, train, test, arguments,n=10,datasetname='500N-KPCrowd'):
	print(f'evaluating {name}')
    dataset = Dataset(datasetname)
    texts_labels = dataset.get_texts()
    train([t[0] for t in texts_labels], arguments=arguments, lang='english')

    predictions = []

    for (text, targets) in tqdm(dataset, ncols=100):
        predictions.append(test(text, arguments=arguments, n=n))

    ap_metrics = mean_ap([t[1] for t in texts_labels], predictions)
    f1_metrics = mean_f1([t[1] for t in texts_labels], predictions)

    print(f"AP scores {name}:")
    for key in ap_metrics:
        print(f"\t{key}: {ap_metrics[key]}")

    print()
    print(f"F1 scores {name}:")
    for key in f1_metrics:
        print(f"\t{key}: {f1_metrics[key]}")
    
    

    with open('evaluations.csv', mode='w+') as csv_file:
    	csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    	csv_writer.writerow( [name] + list(ap_metrics.values()))

if __name__ == "__main__":

    methods = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
       "--tfidf",
       action="store",
       help="train tfidf",
       nargs='*',	
       )
    
    parser.add_argument(
       "--n",
       action="store",
       help="train tfidf",
       default=10,)
    
    parser.add_argument(
       "--dataset",
       action="store",
       help="train tfidf",
       default='500N-KPCrowd'
   )

    args = parser.parse_args()

    if not args.tfidf is None:
    	methods.append(('tfidf',
    		tfidf.train,
    		tfidf.test,
    		args.tfidf,
    		args.n,
    		args.dataset)
    	)

    for m in methods:
        train_method(*m)
