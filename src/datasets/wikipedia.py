from datasets import BaseDataset
from functools import reduce
from itertools import compress

import pandas as pd
import torch


def file_processor(comments_path, annotations_path, demographics_path, task, group_by_gender, percentage, only_male_female, text_processor):
    comments = pd.read_csv(comments_path, sep='\t')[['rev_id', 'comment', 'split']]
    annotations = pd.read_csv(annotations_path, sep='\t')[['rev_id', 'worker_id', f'{task}']]
    demographics = pd.read_csv(demographics_path, sep='\t')[['worker_id', 'gender']]

    # join annotations and genders into dataframe
    comments_annotations = pd.merge(comments, annotations, how='right', on='rev_id')
    data = pd.merge(comments_annotations, demographics, how='left', on='worker_id')

    # filter out NaNs, which can be in gender column
    data = data[data['gender'].notnull()]

    # filter for male and female gender only
    if only_male_female:
        data = data[data['gender'] != 'other']

    # rename columns
    annotator_column = 'worker_id'
    if group_by_gender:
        annotator_column = 'gender'
    data = data.rename(columns={
        'comment': 'text',
        f'{annotator_column}': 'annotator',
        f'{task}': 'label'
    })

    # rename data splits
    data.loc[data['split'] == 'dev', 'split'] = 'validation'

    # only use a part of this dataset
    data_parts = []
    for split in ['train', 'validation', 'test']:
        for annotator in data.annotator.unique().tolist():
            split_idx = int(data.loc[(data['split'] == split) & (data['annotator'] == annotator), :].shape[0] * percentage)
            data_parts.append(data.loc[(data['split'] == split) & (data['annotator'] == annotator), :]
                              .iloc[:split_idx, :])
    data = pd.concat(data_parts, ignore_index=True)
    # also filter comments
    rev_ids = data.rev_id.unique().tolist()
    comments_parts = []
    for rev in rev_ids:
        comments_parts.append(comments.loc[comments['rev_id'] == rev, :])
    comments = pd.concat(comments_parts, ignore_index=True)

    # embed comments only to save memory
    comments['embedding'] = None
    for index, row in comments.iterrows():
        comments.at[index, 'embedding'] = pre_text_processor(row['comment'], text_processor)

    data['pseudo_labels'] = None

    return data, comments


def pre_text_processor(text, text_processor):
    """
    Since the text in the Wikipedia dataset was preprocessed, this function reverses the preprocessing.
    ---- Mapping ----
    NEWLINE_TOKEN: \n
    TAB_TOKEN: \tand discard one annotator 
    `: \"
    """
    text_reversed = text.replace('NEWLINE_TOKEN', '\n').replace('TAB_TOKEN', '\t').replace('`', '\"')
    return text_processor(text_reversed)


class WikipediaDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)

        self.task = args.get('task', 'aggression')
        self.group_by_gender = args.get('group_by_gender', False)
        self.percentage = args.get('percentage', 0.2)
        self.only_male_female = args.get('only_male_female', False)

        self.tasks = ['aggression', 'attack', 'toxicity']
        if self.task not in self.tasks:
            print(f'ERROR - task needs to be one of {self.tasks}')
            return

        root = f'{self.root_data}wikipedia/{self.task}'
        comments_path = f'{root}/{self.task}_annotated_comments.tsv'
        annotations_path = f'{root}/{self.task}_annotations.tsv'
        demographics_path = f'{root}/{self.task}_worker_demographics.tsv'

        self.data, self.comments = file_processor(comments_path, annotations_path, demographics_path,
                                                  self.task, self.group_by_gender, self.percentage, self.only_male_female,
                                                  self.text_processor)

        self.annotators = self.data.annotator.unique().tolist()
        self.data = self.data.to_dict('records')

        self.data_shuffle(split_included=True)

        self.pseudo_labels_key = 'pseudo_labels'

    def __getitem__(self, idx):
        if self.annotator_filter is not '':
            datapoint = [x for x in compress(self.data[self.mode], self.data_mask)][idx]
        else:
            datapoint = self.data[self.mode][idx]

        # get embedding by rev_id
        embedding = self.comments[self.comments['rev_id'] == datapoint['rev_id']]['embedding'].iloc[0]

        # convert to torch tensor
        out = datapoint.copy()
        out['embedding'] = torch.tensor(embedding, device=self.device, dtype=torch.float32)
        out['label'] = torch.tensor(int(datapoint['label']), device=self.device, dtype=torch.long)
        if datapoint['pseudo_labels'] is None:
            out['pseudo_labels'] = {}
        for pseudo_ann in out['pseudo_labels'].keys():
            out['pseudo_labels'][pseudo_ann] = torch.tensor(int(datapoint['pseudo_labels'][pseudo_ann]), device=self.device, dtype=torch.long)

        return out

    def create_pseudo_labels(self, annotator, pseudo_annotator, model):
        # label each data point labeled by annotator with pseudo labels by pseduo_annotator / the model
        for mode in self.data.keys():
            for point in self.data[mode]:
                if point[self.pseudo_labels_key] is None:
                    point[self.pseudo_labels_key] = {}
                if point['annotator'] is annotator and pseudo_annotator not in point[self.pseudo_labels_key].keys():
                    embedding = self.comments[self.comments['rev_id'] == point['rev_id']]['embedding'].iloc[0]
                    inp = torch.tensor(embedding, device=self.device, dtype=torch.float32)
                    pseudo_label = model(inp).argmax().cpu().numpy().item()
                    point[self.pseudo_labels_key][pseudo_annotator] = pseudo_label

        if self.annotator_filter is not '':
            self.data_mask = [x['annotator'] == self.annotator_filter for x in self.data[self.mode]]
