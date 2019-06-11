# Datasets

### Requirements

This class requires the *ake-datasets* repository, which contains a parsed
collection of keyword extraction datasets. To download it, in thsi folder run:

```
$ git clone https://github.com/boudinfl/ake-datasets.git
```

Watch out, it will download around 11 GB of data.

### Usage

In order to use the Dataset class, import it by giving it as its argument
the name of the dataset you want to use. For now, the only ones supported
are `500N-KPCrowd` (500 samples), `DUC-2001` (308 samples), and `Inspec`
(2000 samples).

Example usage from the root folder:

```python
from datasets.datasets import Dataset

# initialize dataset class and get texts and labels
ds = Dataset('DUC-2001')
texts, labels = ds.get_texts()

# this will print out the 10th text in the collection
print(texts[10])

# this will print the list of reference keywords for the 10th text
print(labels[10])
```