from tfidf import tfidf
import pandas as pd
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1, average_precision, f1
import argparse


def train_method(name,train,test,arguments):
	dataset = Dataset('DUC-2001')
	train([t[0] for t in dataset.get_texts()],arguments=arguments, lang='english')

	for (text, targets) in dataset:
		print(f"Targets: {targets}")
		predictions = test(text, arguments=arguments, n=len(targets))
		print(f"Predictions: {predictions}")
		print(f"F1 score: {f1(predictions, targets)}")
		print(f"AP score: {average_precision(predictions, targets)}")
		print()


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
    	methods.append(('tfidf',tfidf.train,tfidf.test,args.tfidf))

    for m in methods:
    	train_method(*m)