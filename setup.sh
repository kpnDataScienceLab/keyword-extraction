pip install -r requirements.txt
python -m spacy download en-vectors-web-lg
pip install git+https://github.com/boudinfl/pke.git
python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
python -m spacy download en 
wget http://vectors.nlpl.eu/repository/11/39.zip