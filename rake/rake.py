from rake_nltk import Rake
import string
import nltk


def train(dataset, arguments, lang='dutch'):
    pass


def test(text,
         arguments,
         k=-1,
         lang='dutch',
         punctuation=False):

    # get list of dutch stopwords
    stopwords = nltk.corpus.stopwords.words(lang)

    r = Rake(
        language=lang,
        stopwords=stopwords,
        max_length=3
    )

    if not punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()

    return keywords[0:k] if len(keywords) >= k else keywords