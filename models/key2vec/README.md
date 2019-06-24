# Key2vec

This model is currently incomplete. All steps that make up the model are
reported both in the original
[paper](additional_resources/papers/key2vec_2018.pdf) and in a condensed
[list](models/key2vec/plan.txt).

## Requirements

This model has two requirements.

#### Fasttext Embeddings

Since the method relies on cosine similarity between embeddings, the paper
calls for using Fasttext embeddings, which are preferable over other
embeddings because they should contain syntactical along with semantic
information.

Right now the current pipeline for obtaining them is as follows:

1. In the [fasttext/embeddings](models/key2vec/fasttext/embeddings) folder,
download and extract the text version of the fasttext embeddings
from the fasttext [website](https://fasttext.cc/docs/en/crawl-vectors.html).
You should end up with a file called `cc.nl.300.vec` which contains 2 million
300-dimensional dutch word embeddings.

2. In order to work with a smaller subset of the emebeddings, to make it
easier on the machine, I wrote a script that should take the set of all
words that occur in the dataset used in the project, and extract only the
fasttext embeddings used in the dataset. Those will be stored in the
[embeddings](models/key2vec/embeddings) folder as a pickled dictionary.

To use the `make_fasttext_subset.py` script you will need to adapt it by
changing the path and/or reading mode of the dataset in the function
`get_data_vocabulary()`.

#### Topic Texts

This model relies on a so-called topic vector. This is the embedding
representation of the topic of the text.

In the original paper this was computed from the title or abstract of the
scientific article to be processed. For now I was using topic texts that
Yunuscan provided me with.

Since this model requires this additional variable, currently the `test()`
function does not conform to the prototype like all other methods.
This will need some fixing.

## Current State

* Step 1 in the step list hasn't been completely implemented. While the
paper calls for Named Entity Recognition to be used for candidate keyword
selection, I am currently only taking n-grams.

* Step 3 in the step list has been adapted. In the paper they call for a
sentence embedding model, so that we can produce an embedding for a keyphrase
and not only for a keyword. Right now, that's being done by averaging the
embeddings of the words in the keyphrase.

* Current progress ends at step 7 in the step list. Code to produce the
adjacency matrix was copy-pasted from stackoverflow (url in the step list
file), but it wasn't adapted or reviewed. All remaining steps should go
as per the list.