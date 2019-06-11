import argparse
import json
import re
import os
import xml.etree.ElementTree as ElementTree


class Dataset:

    def __init__(self, ds_name='500N-KPCrowd'):
        # make sure the dataset is supported (mainly because of the label loading)
        assert ds_name in ['500N-KPCrowd']

        self.ds_name = ds_name
        self.folder_name = 'ake-datasets/datasets/'
        self.ds_folder = self.folder_name + self.ds_name

    def load_labels(self):

        labels = {}
        if self.ds_name == '500N-KPCrowd':
            # load unstemmed labels
            with open(self.ds_folder + '/references/test.reader.json') as handle:
                test_labels = json.load(handle)
            with open(self.ds_folder + '/references/train.reader.json') as handle:
                train_labels = json.load(handle)

            # merge all labels
            labels = {**test_labels, **train_labels}
        return labels

    def get_texts(self):
        """
        Processes all xml files in a folder from the ake-datasets colection and returns a dictionary
        with a text for each file
        :return: Two lists containing all texts and labels
        """
        texts = []

        # load labels in order to return them matched with the texts
        labels_dict = self.load_labels()
        labels = []

        # loop through all documents in the training set
        for (dirpath, dirnames, filenames) in os.walk(self.ds_folder):

            # names of folders in the ake-datasets collection which contain data
            data_folders = ['train', 'dev', 'test']

            # if this folder doesn't contain texts, skip it
            if os.path.basename(dirpath) not in data_folders:
                continue

            for fname in filenames:

                # check that it's an xml file
                if not fname.endswith('.xml'):
                    continue

                text = self.parse_xml(dirpath + '/' + fname)
                text = re.sub(r" 's", "'s", text)
                text = re.sub(r" n't", "n't", text)

                ftitle = re.sub(r'.xml', '', fname)
                if ftitle in labels_dict:
                    texts.append(text)

                    # flatten keyword list (which is currently a list of single-element lists)
                    labels.append([keyword for sublist in labels_dict[ftitle] for keyword in sublist])

        return texts, labels

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ds_name",
        type=str,
        default='500N-KPCrowd',
        choices=['500N-KPCrowd'],
        help="Name of the dataset to use"
    )

    flags = parser.parse_args()
    ds = Dataset(flags.ds_name)