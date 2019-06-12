Written by Arvid L. June 11th

To run anything using pke:

pip install git+https://github.com/boudinfl/pke.git

# Requirements
python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
# download the english model
python -m spacy download en 

# See file pke_dutchTV_test.py for an example of how 
# to use pke loaded models