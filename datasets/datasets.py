import json
import re
import os
import xml.etree.ElementTree as ElementTree
import pandas as pd


class Dataset:

    def __init__(self, ds_name='500N-KPCrowd'):
        self.ds_name = ds_name
        self.texts = []
        self.labels = []

        if ds_name.endswith('.csv'):
            # assume ds_name is the path of the dataset file from the dataset folder
            self.build_csv_dataset(ds_path=ds_name)
        else:
            # load the dataset from the ake-datasets repository that was cloned in the datasets folder
            self.ake_folder_name = os.path.dirname(os.path.realpath(__file__)) + '/ake-datasets/datasets/'
            self.ake_ds_folder = self.ake_folder_name + self.ds_name
            self.build_ake_dataset()

    def __len__(self):
        return len(self.texts)

    def load_ake_labels(self):

        labels_path = self.ake_ds_folder + '/references/'

        labels = {}
        label_files = []
        if self.ds_name == 'dutch_sub':
            label_files = ['test.reader.json']
        if self.ds_name == '500N-KPCrowd':
            label_files = ['test.reader.json', 'train.reader.json']
        elif self.ds_name == 'DUC-2001':
            label_files = ['test.reader.json']
        elif self.ds_name == 'Inspec':
            label_files = ['dev.contr.json', 'test.contr.json', 'train.contr.json']
        elif self.ds_name in ['NUS', 'WWW']:
            label_files = ['test.combined.json']
        elif self.ds_name == 'KDD':
            label_files = ['test.author.json']
        else:
            raise ValueError('This dataset doesn\'t exist in the ake-datasets repository. Refer to '
                             'https://github.com/boudinfl/ake-datasets/tree/master/datasets to know which '
                             'datasets to use.')

        for lfile in label_files:
            # load unstemmed labels
            with open(labels_path + lfile) as handle:
                l = json.load(handle)
                labels = {**labels, **l}

        return labels

    def build_ake_dataset(self):
        """
        Processes all xml files in a folder from the ake-datasets colection and returns a dictionary
        with a text for each file
        :return: Two lists containing all texts and labels
        """

        # load labels in order to return them matched with the texts
        labels_dict = self.load_ake_labels()

        # loop through all documents in the training set
        for (dirpath, dirnames, filenames) in os.walk(self.ake_ds_folder):

            # names of folders in the ake-datasets collection which contain data
            data_folders = ['train', 'dev', 'test']

            # if this folder doesn't contain texts, skip it
            if os.path.basename(dirpath) not in data_folders:
                continue

            for fname in filenames:

                # check that it's an xml file
                if fname.endswith('.xml'):
                    

                    text = self.parse_xml(dirpath + '/' + fname)
                    text = self.clean_text(text)

                    ftitle = re.sub(r'.xml', '', fname)
                    if ftitle in labels_dict:
                        self.texts.append(text)
                        self.labels.append([keyword for sublist in labels_dict[ftitle] for keyword in sublist])
                
                if fname.endswith('.txt'):
                    with open(dirpath+'/'+fname,'r',encoding='latin-1') as f:
                        f.seek(0)
                        self.texts.append(f.read())
                        self.labels.append([fname])


    def build_csv_dataset(self, ds_path):
        full_path = os.path.dirname(os.path.realpath(__file__)) + '/' + ds_path
        dataframe = pd.read_csv(full_path)

        for index, row in dataframe.iterrows():
            if type(row['text']) is str:
                self.texts.append(row['fixed_text'])

                # the labels in the labels column should be separated by the | character
                self.labels.append('|'.split(row['labels']))

    @staticmethod
    def clean_text(text):
        text = re.sub(r" 's", "'s", text)
        text = re.sub(r" n't", "n't", text)
        text = re.sub(r" 're", "'re", text)
        text = re.sub(r" 'll", "'ll", text)
        text = re.sub(r"'' `` | '' ''", '"', text)
        text = re.sub(r"`` | ''", '"', text)
        text = re.sub(r" -LRB- ", " (", text)
        text = re.sub(r" -RRB- ", ") ", text)
        text = re.sub(r" -RRB- ", ") ", text)
        text = re.sub(r" -LSB- ", " [", text)
        text = re.sub(r" -RSB- ", "] ", text)

        # Tags without spaces:
        text = re.sub(r"-LRB-", " (", text)
        text = re.sub(r"-RRB-", ") ", text)
        text = re.sub(r"-RRB-", ") ", text)
        text = re.sub(r"-LSB-", " [", text)
        text = re.sub(r"-RSB-", "] ", text)

        # Unkown tags: 
        text = re.sub(r"-RCB-", "", text)
        text = re.sub(r"-LCB-", "", text)

        return text

    @staticmethod
    def parse_xml(file_path):
        """
        Takes in an xml file from the ake-datasets collection and stitches together the token list
        :param file_path: The file path of the xml file
        :return: The full text that has been stitched together
        """

        # parse the xml tree structure
        tree = ElementTree.parse(file_path)
        root = tree.getroot()

        # get the text for the current document
        tokens = [tok for doc in root for sents in doc for sent in sents for toks in sent for tok in toks]
        return ' '.join([tok[0].text for tok in tokens])

    def __iter__(self):
        for text, labels in zip(self.texts, self.labels):
            yield text, labels
