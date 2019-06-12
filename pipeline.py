from tfidf.tfidf import train,test
import pandas as pd
from datasets.datasets import Dataset
from eval_metrics import mean_ap, mean_f1, average_precision, f1

dataset = Dataset('DUC-2001')
train([t[0] for t in dataset.get_texts()])

for (text, target) in dataset:
	print(f"Targets: {target}")
	predictions = test(text)
	print(f"Predictions: {predictions}")
	print(f"F1 score: {f1(predictions, target)}")
	print(f"AP score: {average_precision(predictions, target)}")
	print()
