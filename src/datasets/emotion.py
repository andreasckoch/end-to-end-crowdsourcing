from datasets import BaseDataset
from functools import reduce
from itertools import compress

import pandas as pd
import torch


def file_processor(path, text_processor):
    data = pd.read_csv(path, sep='\t').rename(columns={"headline": "text"})

    data['embedding'] = None
    for index, row in data.iterrows():
        data.at[index, 'embedding'] = text_processor(row['text'])

    return data


def emotion_file_processor(path, emotion):
    data = pd.read_csv(path, sep='\t').rename(
        columns={'gold': f'{emotion}_gold', 'response': f'{emotion}_response', '!amt_worker_ids': 'annotator'})

    data[f'{emotion}_label'] = None
    data[f'{emotion}_pseudo_labels'] = None
    for index, row in data.iterrows():
        data.at[index, f'{emotion}_label'] = encode_scores(row[f'{emotion}_response'])

    return data


def encode_scores(score, maximum_value=100, starting_value=-100, num_of_classes=3, separate_zero_class_idx=1):
    """
    Ranges for all emotions: [0, 100]
    Exception is 'valence' with: [-100, 100]
    """
    step = maximum_value * 2 / num_of_classes
    ranges = [{'start': starting_value + i * step, 'end': starting_value + (i + 1) * step} for i in range(num_of_classes)]
    if separate_zero_class_idx is not None:
        ranges[separate_zero_class_idx] = {'start': 0, 'end': 0}
        ranges[separate_zero_class_idx - 1] = {'start': ranges[separate_zero_class_idx - 1]['start'], 'end': 0}
        ranges[separate_zero_class_idx + 1] = {'start': 0, 'end': ranges[separate_zero_class_idx + 1]['end']}
        if score is 0:
            return separate_zero_class_idx

    for idx, ran in enumerate(ranges):
        if score >= ran['start'] and score <= ran['end']:
            return idx


class EmotionDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)

        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'valence']
        self.emotion = 'valence'

        root = f'{self.root_data}emotion'
        path = f'{root}/affect.tsv'

        affect = file_processor(path, self.text_processor)

        data = []
        for emotion in self.emotions:
            path = f'{root}/{emotion}.standardized.tsv'
            data.append(emotion_file_processor(path, emotion))

        emotions = reduce(lambda left, right: pd.merge(
            left, right, on=['!amt_annotation_ids', 'annotator', 'orig_id']), data)

        self.data = pd.merge(emotions, affect, how='left', left_on='orig_id', right_on='id')
        
        ds_experiment = args.get('ds_experiment', False)
        if ds_experiment:
            ds_labels = pd.read_csv(f'{root}/fds_generated_labels.tsv', sep='\t').rename(columns={"label": "ds_label"})
            self.data = pd.merge(self.data, ds_labels, how='left', left_on='id', right_on='id')
            self.emotion = 'ds'

        self.annotators = self.data.annotator.unique().tolist()

        self.data = self.data.to_dict('records')

        # do custom split
        self.custom_data_split()
        no_shuffle = args.get('no_shuffle', False)
        if no_shuffle is False:
            self.data_shuffle_after_split()

        self.pseudo_labels_key = f'{self.emotion}_pseudo_labels'

    def set_emotion(self, emotion):
        if emotion not in self.emotions + ['ds']:
            raise Exception(f"Emotion must be one of these: \n{','.join(self.emotions)}")
        self.emotion = emotion
        self.pseudo_labels_key = f'{self.emotion}_pseudo_labels'

    def custom_data_split(self):
        # since split isn't always the same for some reason, do it explicitly here
        texts = set([point['text'] for point in self.data])
        modes = ['train', 'validation', 'test']
        split_at = {'train': 72, 'validation': 18, 'test': 10}
        new_texts = {'train': [], 'validation': [], 'test': []}
        annotator_texts = {}
        for ann in self.annotators:
            annotator_texts[ann] = [point['text'] for point in self.data if point['annotator'] == ann]
        problem_text = "Outcry at N Korea 'nuclear test'"
        for mode in modes:
            i = 0
            while len(new_texts[mode]) < split_at[mode]:
                if len(annotator_texts[self.annotators[i]]) > 0:
                    text = annotator_texts[self.annotators[i]].pop(0)
                    # if text == problem_text:
                    #     print(f'{annotator_texts[self.annotators[i]]}')
                    new_texts[mode].append(text)
                    for ann in self.annotators:
                        if text in annotator_texts[ann]:
                            annotator_texts[ann].remove(text)
                            # if text == problem_text:
                            #     print(f'{annotator_texts[ann]}')
                i = (i + 1) % len(self.annotators)

        new_data = {
            mode: [point for point in self.data for text in new_texts[mode] if point['text'] == text]
            for mode in modes
        }
        self.data = new_data

    def __getitem__(self, idx):
        if self.annotator_filter is not '':
            datapoint = [x for x in compress(self.data[self.mode], self.data_mask)][idx]
        else:
            datapoint = self.data[self.mode][idx]

        # convert to torch tensor
        out = datapoint.copy()
        out['embedding'] = torch.tensor(datapoint['embedding'], device=self.device, dtype=torch.float32)
        out['label'] = torch.tensor(int(datapoint[f'{self.emotion}_label']), device=self.device, dtype=torch.long)

        if (self.pseudo_labels_key not in datapoint) or datapoint[self.pseudo_labels_key] is None:
            out[self.pseudo_labels_key] = {}
        out['pseudo_labels'] = {}
        for pseudo_ann in out[self.pseudo_labels_key].keys():
            out['pseudo_labels'][pseudo_ann] = torch.tensor(int(datapoint[self.pseudo_labels_key][pseudo_ann]), device=self.device, dtype=torch.long)

        return out
    
    def data_shuffle(self, split_included=False):
        import random
        random.seed(123456789)
        random.shuffle(self.data)
        
        headlines = []
        for datapoint in self.data:
            headlines.append(datapoint['text'])
        headlines = list(set(headlines))
        random.shuffle(headlines)

        length = 100
        eof_train_split = int(length * self.train_val_split * 0.9)
        eof_val_split = int(length * 0.9)
        
        train_filter = [(x['text'] in headlines[0:eof_train_split]) for x in self.data] 
        valid_filter = [(x['text'] in headlines[eof_train_split:eof_val_split]) for x in self.data] 
        test_filter  = [(x['text'] in headlines[eof_val_split:]) for x in self.data]

        self.data = {
            'train': [x for x in compress(self.data, train_filter)],
            'validation': [x for x in compress(self.data, valid_filter)],
            'test': [x for x in compress(self.data, test_filter)]
        }

        if self.annotator_filter is not '':
            self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]
