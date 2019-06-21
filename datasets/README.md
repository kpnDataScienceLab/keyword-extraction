# Datasets

## Requirements

NOTE: this should automatically be done by running setup script in
the [root](..) folder.

This class requires the [ake-datasets](https://github.com/boudinfl/ake-datasets)
repository, which contains a parsed collection of keyword extraction datasets.
To download it, clone it in this folder with:

```
$ git clone https://github.com/boudinfl/ake-datasets.git
```

Watch out, it will download around 11 GB of data.

## Usage

In order to use the Dataset class, import it by giving it as its argument
the name of the dataset you want to use, either from the
[available datasets](#available-datasets) or by using the path of a
[custom dataset](#custom-datasets).

Example usage from the root folder:

```python
from datasets.datasets import Dataset

# initialize dataset class and get texts and labels
ds = Dataset('DUC-2001')

# this will print out the 10th text in the collection
print(ds.texts[10])

# this will print the list of reference keywords for the 10th text
print(ds.labels[10])
```

## Available Datasets

Currently, all datasets in this table are available.

| dataset                | lang | nature       | train | dev | test | Annotation  | #kp (test) | #words (test) |
| ---------------------- | ---- | ------------ | ----: | --: | ---: | ----------: | ---------: | ------------: |
| NUS [1]                | en   | Full papers  | -     | -   | 211  | A+R         | 11.0       | 8398.3        |
| Inspec [2]             | en   | Abstracts    | 1000  | 500 | 500  | I (uncontr) | 9.8        | 134.6         |
| KDD [3]                | en   | Abstracts    | -     | -   | 755  | A           | 4.1        | 190.7         |
| WWW [3]                | en   | Abstracts    | -     | -   | 1330 | A           | 4.8        | 163.5         |
| DUC-2001 [4]           | en   | News         | -     | -   | 308  | R           | 8.1        | 847.2         |
| 500N-KPCrowd [5]       | en   | News         | 450   | -   | 50   | R           | 46.2       | 465.3         |

Where annotations were produced by authors (A), readers (R) or professional
indexers (I).

## Custom Datasets

You can use any custom dataset that is in CSV format. The file should contain
all texts in a column named `text`, and all labels in a column named `labels`.
The file may be placed anywhere inside this directory.

The labels for a text are supposed to be a list of keyphrases, which in the
CSV file should appear as a single string where the keyphrases are
separated by the pipe `|` character. For instance, one entry in the label
column could look like:

```
dog|walk in the park|sun|ice-cream|hot weather
```

The custom dataset can then be loaded using the `Dataset` class by using:

```python
ds = Dataset('file/path.csv')
```

Where the file path is relative to this directory.

## References

1. **Keyphrase Extraction in Scientific Publications.**
   Thuy Dung Nguyen and Min-Yen Kan.
   *In Proceedings of International Conference on Asian Digital Libraries 2007.*
   p. 317-326.
   
2. **Improved automatic keyword extraction given more linguistic knowledge.**
   Anette Hulth.
   *In Proceedings of EMNLP 2003.*
   p. 216-223.
   
3. **Citation-Enhanced Keyphrase Extraction from Research Papers: A Supervised
   Approach.**
   Cornelia Caragea, Florin Bulgarov, Andreea Godea and Sujatha Das Gollapalli.
   *In Proceedings of EMNLP 2014.*
   pp. 1435-1446.
   
4. **Single Document Keyphrase Extraction Using Neighborhood Knowledge.**
   Xiaojun Wan and Jianguo Xiao.
   *In Proceedings of AAAI 2008.*
   pp. 855-860.

5. **Supervised Topical Key Phrase Extraction of News Stories using
   Crowdsourcing, Light Filtering and Co-reference Normalization.**
   Marujo, L., Gershman, A., Carbonell, J., Frederking, R., & Neto, J. P.
   *In Proceedings of LREC 2012.*