from datasets import BaseDataset
from functools import reduce

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
    for index, row in data.iterrows():
        data.at[index, f'{emotion}_label'] = encode_scores(row[f'{emotion}_response'])
        
    return data


def encode_scores(score, maximum_value=100, num_of_classes=2):
    return int(score*num_of_classes/100)


class EmotionDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)
        
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'valence']
        self.emotion = 'anger'

        root = f'{self.root_data}emotion'
        path = f'{root}/affect.tsv'
        
        affect = file_processor(path, self.text_processor)

        data = []
        for emotion in self.emotions:
            path = f'{root}/{emotion}.standardized.tsv'
            data.append(emotion_file_processor(path, emotion))

        emotions = reduce(lambda left,right: pd.merge(
            left, right, on=['!amt_annotation_ids', 'annotator', 'orig_id']), data)

        self.data = pd.merge(emotions, affect, how='left', left_on='orig_id', right_on='id')
        self.data = self.data.to_dict('records')

        self.data_shuffle()

    def set_emotion(self, emotion):
        if emotion not in self.emotions:
            raise Exception(f"Emotion must be one of these: \n{','.join(self.emotions)}")
        self.emotion = emotion

    def __getitem__(self, idx):
        if self.annotator_filter is not '':
            datapoint = [x for x in compress(self.data[self.mode], self.data_mask)][idx]
        else:
            datapoint = self.data[self.mode][idx]

        # convert to torch tensor
        out = datapoint.copy()
        out['embedding'] = torch.tensor(datapoint['embedding'], device=self.device, dtype=torch.float32)
        out['label'] = torch.tensor(int(datapoint[f'{self.emotion}_label']), device=self.device, dtype=torch.long)

        return out