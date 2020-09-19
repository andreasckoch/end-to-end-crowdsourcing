import torch
import numpy as np
from itertools import compress

from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    """Dataset Template"""

    def __init__(self, **argv):
        self.argv = argv

        self.mode = 'train'
        self.annotator_filter = ''
        self.train_val_split = argv.get('train_val_split', 0.8)
        self.device = argv.get('device', torch.device('cpu'))

        self._build_text_processor(**argv)
        pass

    def _build_text_processor(self, **argv):
        text_processor = argv.get('text_processor', 'word2vec').lower()
        text_processor_filters = argv.get('text_processor_filters', [])

        if text_processor == 'word2vec':
            from datasets.processors.word2vec import _build_text_processor, text_processor
            self.text_processor_model = _build_text_processor(**argv)
            self.text_processor_func = text_processor

        self.text_processor_filters = []
        for f in text_processor_filters:
            if f == 'stopwordfilter' or f == 'stopwordsfilter':
                from datasets.transformers.text import stopwordsfilter
                self.text_processor_filters.append(stopwordsfilter)

            if f == 'lowercase' or f == 'lower':
                from datasets.transformers.text import lowercase
                self.text_processor_filters.append(lowercase)

    def text_processor(self, text, **argv):
        for _filter in self.text_processor_filters:
            text = _filter(text)
            if text is '':
                # TODO: maybe exclude these samples? Right now we get all zeros from text_processor_func
                pass
        return self.text_processor_func(self.text_processor_model, text, **argv)

    def data_shuffle(self):
        import random
        random.seed(123456789)
        random.shuffle(self.data)

        length = len(self.data)
        eof_train_split = int(length * self.train_val_split * 0.9)
        eof_val_split = int(length * 0.9)

        self.data = {
            'train': self.data[0:eof_train_split],
            'validation': self.data[eof_train_split:eof_val_split],
            'test': self.data[eof_val_split:]
        }

        if self.annotator_filter is not '':
            self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]

    def set_mode(self, mode):
        if mode not in ['train', 'validation', 'test']:
            raise Exception('mode must be train or validation or test')
        self.mode = mode
        if self.annotator_filter is not '':
            self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]

    def set_annotator_filter(self, annotator_filter):
        self.annotator_filter = annotator_filter
        self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]

    def no_annotator_filter(self):
        self.annotator_filter = ''
        self.data_mask = None

    def __len__(self):
        if self.annotator_filter is not '':
            return len([x for x in compress(self.data[self.mode], self.data_mask)])
        else:
            return len(self.data[self.mode])

    def __getitem__(self, idx):
        if self.annotator_filter is not '':
            datapoint = [x for x in compress(self.data[self.mode], self.data_mask)][idx]
        else:
            datapoint = self.data[self.mode][idx]

        # convert to torch tensor
        out = datapoint.copy()
        out['embedding'] = torch.tensor(datapoint['embedding'], device=self.device, dtype=torch.float32)
        out['label'] = torch.tensor(int(datapoint['label']), device=self.device, dtype=torch.long)

        return out


class SimpleCustomBatch:
    """
    Create function to apply to batch to allow for memory pinning when using a custom batch/custom dataset.
    Following guide on https://pytorch.org/docs/master/data.html#single-and-multi-process-data-loading
    """

    def __init__(self, data, device):
        self.input = torch.stack([sample['embedding'] for sample in data]).to(device=device)
        self.target = torch.stack([sample['label'] for sample in data]).to(device=device)

    def pin_memory(self):
        self.input = self.input.pin_memory()
        self.target = self.target.pin_memory()
        return self


def collate_wrapper(batch, device=torch.device('cuda')):
    return SimpleCustomBatch(batch, device)
