from tfidf.tfidf import train,test
import pandas as pd
from datasets.datasets import Dataset
from eval_metrics import mean_ap, f1

dataset = Dataset()
train(dataset.get_texts())

for (text,target) in dataset:
	result = test(text)
	print(f_1(result,target))