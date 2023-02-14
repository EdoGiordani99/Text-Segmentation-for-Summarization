import torch
import pandas as pd

from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class SegmentationDataset(Dataset):

    def __init__(self, texts_folder: str, annotations_path: str):
        """
        Args:
            data_path (str): folder of texts
            annotations_path (str): path to the annotation file
        """

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        dataframe = pd.read_csv(annotations_path, sep='\t')
        # [TEXT ID, END INTRO, END DEV]
        self.annots = dataframe.values.tolist()

        self.dataset = {'texts': [], 'embeds': [], 'labels': []}

        self.create_dataset(texts_folder)

        self.pad_data()

        self.embed_texts()

    def __getitem__(self, idx):
        return {'texts': self.dataset['texts'][idx],
                'embeds': self.dataset['embeds'][idx],
                'labels': self.dataset['labels'][idx]}

    def __len__(self):
        return len(self.dataset['labels'])

    def split_index_and_sent(self, s):
        """
        If string is '34: Hello World', it returns:
        [34, 'Hello World']
        """
        stop = 0
        for i in range(1, 10):
            b = s[:i]
            try:
                int(b)
            except:
                stop = i
                break

        return [int(s[:stop - 1]), s[stop + 1:]]

    def create_dataset(self, texts_folder):
        """
        Reads texts and associates sentences with annotations
        """
        not_found = []

        for i, ann in enumerate(self.annots[:4]):

            sents, labs = [], []

            text_id, end_intro, end_dev = ann
            text_path = texts_folder + '/' + text_id + '.txt'

            try:
                with open(text_path) as f:
                    text = f.read()
            except:
                not_found.append(text_path)
                break

            # For each sentence
            for sentence in text.split('\n'):
                id, sent = self.split_index_and_sent(sentence)

                if id <= end_intro:
                    lab = 1  # Introduction
                elif end_intro < id <= end_dev:
                    lab = 2  # Development
                else:
                    lab = 3  # Conclusion

                sents.append(sent)
                labs.append(lab)

            self.dataset['texts'].append(sents)
            self.dataset['labels'].append(labs)

    def pad_data(self):
        """
        Pad to have all texts with the same number of sentences
        """

        max_len = len(max(self.dataset['labels'], key=len))

        for x, y in zip(self.dataset['texts'], self.dataset['labels']):
            diff = max_len - len(y)

            x.extend(['[PAD]'] * diff)
            y.extend([0] * diff)  # Pad Label

    def embed_texts(self):
        for text in self.dataset['texts']:
            embed = self.embedder.encode(text)
            self.dataset['embeds'].append(embed)


def collate_fn(batch):
    """
    Process the batch
    Args:
        batch (list of dict): [{'texts': [list of sentences],
                                'embeds': [list of sentence embeddings],
                                'labels': [list of sentence labels]}
    Return:
        batch (dict): {'embeds': torch.Tensor([tokenized sentences])
                       'labels': torch.Tensor([sentence labels]),
                       'texts': list of texts (list of sentences)}
    """
    texts = []
    embeds = []
    labs = []

    for el in batch:
        texts.append(el['texts'])
        embeds.append(el['embeds'])
        labs.append(el['labels'])

    return {'texts': texts,
            'embeds': torch.Tensor(embeds),
            'labels': torch.LongTensor(labs)}
