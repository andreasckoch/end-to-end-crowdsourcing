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
        self.root_data = argv.get('data_path', '../data/')

        self.pseudo_labels_key = 'pseudo_labels'

        self._build_text_processor(**argv)
        pass

    def _build_text_processor(self, **argv):
        text_processor = argv.get('text_processor', 'word2vec').lower()
        text_processor_filters = argv.get('text_processor_filters', ['lowercase'])

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

    def data_shuffle(self, split_included=False):
        import random
        random.seed(123456789)
        random.shuffle(self.data)

        if split_included:
            self.data = {
                'train': [point for point in self.data if point['split'] == 'train'],
                'validation': [point for point in self.data if point['split'] == 'validation'],
                'test': [point for point in self.data if point['split'] == 'test']
            }
        else:
            # make sure there are no samples in two splits
            unique_samples = list(set([point['text'] for point in self.data]))
            length = len(unique_samples)
            eof_train_split = int(length * self.train_val_split * 0.9)
            eof_val_split = int(length * 0.9)

            self.data = {
                'train': [point for point in self.data if point['text'] in unique_samples[0:eof_train_split]],
                'validation': [point for point in self.data if point['text'] in unique_samples[eof_train_split:eof_val_split]],
                'test': [point for point in self.data if point['text'] in unique_samples[eof_val_split:]]
            }

            # length = len(self.data)
            # eof_train_split = int(length * self.train_val_split * 0.9)
            # eof_val_split = int(length * 0.9)

            # self.data = {
            #     'train': self.data[0:eof_train_split],
            #     'validation': self.data[eof_train_split:eof_val_split],
            #     'test': self.data[eof_val_split:]
            # }

        if self.annotator_filter is not '':
            self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]

    def data_shuffle_after_split(self):
        import random
        random.shuffle(self.data['train'])
        random.shuffle(self.data['validation'])
        random.shuffle(self.data['test'])
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

    def create_pseudo_labels(self, annotator, pseudo_annotator, model):
        # label each data point labeled by annotator with pseudo labels by pseduo_annotator / the model
        for mode in self.data.keys():
            for point in self.data[mode]:
                if point[self.pseudo_labels_key] is None:
                    point[self.pseudo_labels_key] = {}
                if point['annotator'] is annotator and pseudo_annotator not in point[self.pseudo_labels_key].keys():
                    inp = torch.tensor(point['embedding'], device=self.device, dtype=torch.float32)
                    pseudo_label = model(inp).argmax().cpu().numpy().item()
                    point[self.pseudo_labels_key][pseudo_annotator] = pseudo_label

        if self.annotator_filter is not '':
            self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]

    def remove_pseudo_labels(self):
        for mode in self.data.keys():
            for point in self.data[mode]:
                point[self.pseudo_labels_key] = {}

    def use_custom_labels(self, sample_label_map, mode='train'):
        """Use custom labels and discard redundant training data

        Args:
            sample_label_map (dict): map every sample text to its predicted label
        """
        self.set_mode(mode)
        samples = list(sample_label_map.keys())
        new_data = []
        for point in self.data[self.mode]:
            if point['text'] in samples and point['text'] not in [new_point['text'] for new_point in new_data]:
                point['label'] = sample_label_map[point['text']]
                point['annotator'] = 'custom'
                new_data.append(point)
        self.data[self.mode] = new_data

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
        if datapoint['pseudo_labels'] is None:
            out['pseudo_labels'] = {}
        for pseudo_ann in out['pseudo_labels'].keys():
            out['pseudo_labels'][pseudo_ann] = torch.tensor(int(datapoint['pseudo_labels'][pseudo_ann]), device=self.device, dtype=torch.long)

        return out


class SimpleCustomBatch:
    """
    Create function to apply to batch to allow for memory pinning when using a custom batch/custom dataset.
    Following guide on https://pytorch.org/docs/master/data.html#single-and-multi-process-data-loading
    """

    def __init__(self, data, device):
        self.input = torch.stack([sample['embedding'] for sample in data]).to(device=device)
        self.target = torch.stack([sample['label'] for sample in data]).to(device=device)

        if 'pseudo_labels' in data[0].keys():
            self.pseudo_targets = [sample['pseudo_labels'] for sample in data]
        else:
            self.pseudo_targets = []

        # record annotator information in list
        self.annotations = ([sample['annotator'] for sample in data])

    def pin_memory(self):
        self.input = self.input.pin_memory()
        self.target = self.target.pin_memory()
        return self


def collate_wrapper(batch, device=torch.device('cuda')):
    return SimpleCustomBatch(batch, device)


def collate_wrapper_cpu(batch, device=torch.device('cpu')):
    return SimpleCustomBatch(batch, device)
