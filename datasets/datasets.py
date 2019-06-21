import argparse
import json
import re
import os
import xml.etree.ElementTree as ElementTree


class Dataset:

    def __init__(self, ds_name='500N-KPCrowd'):
        # make sure the dataset is supported (mainly because of the label loading)
        assert ds_name in ['500N-KPCrowd', 'DUC-2001', 'Inspec', 'SemEval-2010', 'NUS', 'WWW', 'KDD','dutch_sub']

        self.ds_name = ds_name
        self.folder_name = os.path.dirname(os.path.realpath(__file__)) + '/ake-datasets/datasets/'
        self.ds_folder = self.folder_name + self.ds_name
        self.texts = []
        self.labels = []
        self.build_dataset()

    def __len__(self):
        return len(self.texts)

    def load_labels(self):

        labels_path = self.ds_folder + '/references/'

        labels = {}
        label_files = []
        if self.ds_name == 'dutch_sub':
            label_files = ['test.reader.json']
        if self.ds_name == '500N-KPCrowd':
            label_files = ['test.reader.json', 'train.reader.json']
        if self.ds_name == 'DUC-2001':
            label_files = ['test.reader.json']
        if self.ds_name == 'Inspec':
            label_files = ['dev.contr.json', 'test.contr.json', 'train.contr.json']
        if self.ds_name == 'SemEval-2010':
            print("[WARNING] SemEval-2010's labels are stemmed!")
            label_files = ['test.combined.stem.json', 'train.combined.stem.json']
        if self.ds_name in ['NUS', 'WWW']:
            label_files = ['test.combined.json']
        if self.ds_name == 'KDD':
            label_files = ['test.author.json']

        for lfile in label_files:
            # load unstemmed labels
            with open(labels_path + lfile) as handle:
                l = json.load(handle)
                labels = {**labels, **l}

        return labels

    def build_dataset(self):
        """
        Processes all xml files in a folder from the ake-datasets colection and returns a dictionary
        with a text for each file
        :return: Two lists containing all texts and labels
        """

        # load labels in order to return them matched with the texts
        labels_dict = self.load_labels()

        # loop through all documents in the training set
        for (dirpath, dirnames, filenames) in os.walk(self.ds_folder):

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
        for text in self.texts:
            yield text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ds_name",
        type=str,
        default='500N-KPCrowd',
        choices=['500N-KPCrowd', 'DUC-2001', 'Inspec', 'SemEval-2010', 'NUS', 'WWW', 'KDD'],
        help="Name of the dataset to use"
    )

    flags = parser.parse_args()
    ds = Dataset()
    t, l = ds.get_texts()
    breakpoint()
