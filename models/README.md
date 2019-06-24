# Keyword Extraction Project - Models

## extending the project
Adding new methods to the pipeline is not difficult.
We have defined a simple API to setup models and test/train them.

To add a new model the following steps should be followed



1. Add that model as a new module in [models](models), making sure to stick
to the requirements mentioned in that folder. The train and test functions should look like this:

```python
def train(dataset, arguments, lang='dutch'):
	"""
	dataset: the dataset is a list of all documents.
	arguments: this is a list of all commandline arguments.
	lang: the language that is used.
	"""

```
```python
def test(text, arguments, k=5, lang='dutch'):
	"""
	text: the text we want to extract the keywords from.
	arguments: this is a list of all commandline arguments.
	k: the amount of keywords that should be returned.
	lang: the language that is used.
	"""
```

2. In `pipeline.py`, import that module from the models folder.

```python
#pipeline.py
from models import new_model
```

3. Add the relevant arguments to `pipeline.py` by including an `argparse` argument
along with the others in the form:
```python
	#line 182
    parser.add_argument(
        "--new_model",
        help="Use the new model",
        nargs='*'
    )
```
```python
	#line 304
        if args.new_model is not None:
            methods.append({'name': 'Ensemble',
                            'train_function': new_model.train,
                            'test_function': new_model.test,
                            'arguments': args.new_model,
                            'k': args.k,
                            'dataset_name': dataset,
                            'match_type': args.matchtype}
                           )

```


