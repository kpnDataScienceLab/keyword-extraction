# Keyword Extraction Project - Models

## extending the project
Adding new methods to the pipeline is not difficult.
We have defined a simple API to setup models and test/train them.
As long as our new method has these two functions all should be good.


```
def train(dataset, arguments, lang='dutch'):
	"""
	dataset: the dataset is a list of all documents.
	arguments: this is a list of all commandline arguments.
	lang: the language that is used
	"""
...

```
```
def test(text, arguments, k=5, lang='dutch'):
	"""
	text: the text we want the keywords from.
	arguments: this is a list of all commandline arguments.
	k: the amount of keywords that should be returned.
	lang: the language that is used.
	"""
...

```

If we have implemented these two functions, we can extend our model in the following manner:
```
#pipeline.py
from models import new_model
.
.
.
#line 182
    parser.add_argument(
        "--new_model",
        help="Use the new model",
        nargs='*'
    )
.
.
.
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


