# Keyword Extraction Project

## Requirements

Joseph show us how it's done ;) for instance:

```
$ python set_me_up_inside.sh
```

## Usage

To test any of the implemented methods against any of the datasets, run:

```
$ python pipeline.py [--mprank] [--positionrank] [--singlerank] [--textrank] [--topicrank]
                     [--yake] [--bm25] [--tfidf] [--rake] [--graphmodel] [--ensemble]
                     [--k] [--dataset] [--matchtype]
```

Where:

* The `--k` flag indicates the maximum number of keywords retrieved per
text, which also affects the evaluation (F1@k). The default is 20.

* The `--dataset` flag takes either the name of a dataset (of which the
supported ones currently are `500N-KPCrowd`, `DUC-2001`, `Inspec`,
`NUS`, `WWW`, `KDD`). Alternatively, it takes the argument `all` which
uses all of the listed datasets. Check [datasets](datasets) for more
information on the supported datasets and using custom ones.

* The `--matchtype` flag takes the type of string comparison to be used
performing any comparison between keyphrases (e.g. during the [evaluation](utils)).
The possibilities are:
    * `strict`: a simple string equality comparison
    * `levenshtein`: a comparison that requires the two keyphrases
    to be at a [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
    of less than 1 from each other
    * `intersect`: a check of whether any word is shared between the two
    keyphrases
    * `spacy`: a comparison using the cosine similarity of the word embeddings
    for the two keyphrases. WARNING: this takes too long to be useful.

The other flags don't require arguments and specify whether to use that
method during the current run. Multiple methods may be used in the same
run.

All methods will then be tested sequentially and the results will be saved
as CSV files in `evaluations/`

## Adding models

In order to add any model to the current pipeline, two steps are required:

1. Add that model as a new module in [models](models), making sure to stick
to the requirements mentioned in that folder.

2. Import the module from `pipeline.py`.

3. Add the relevant arguments to `pipeline.py` by including an `argparse` argument
along with the others in the form:

```python
parser.add_argument(
    "--model_name",
    help="Use model_name",
    nargs="*"  # arguments to the model. will be fed to it as a list of strings
)
```

And by adding the relevant parameters in the loop just below that in the form:

```python
methods.append({'name': 'Model Name',
                'train_function': module_name.train,
                'test_function': module_name.test,
                'arguments': args.model_name,
                'k': args.k,
                'dataset_name': dataset,
                'match_type': args.matchtype}
               )
```