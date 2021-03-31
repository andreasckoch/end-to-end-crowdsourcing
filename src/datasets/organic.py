import pandas as pd

from datasets import BaseDataset


def file_processor(path, text_processor, split, sep='|', predict_coarse_attributes_task=False, entity_filter='organic'):
    data = pd.read_csv(path, sep=sep)[
        ['Sentiment', 'Entity', 'Attribute', 'Sentence', 'Annotator']]
    data = data.rename(columns={
        'Sentiment': 'sentiment',
        'Entity': 'entity',
        'Attribute': 'attribute',
        'Sentence': 'text',
        'Annotator': 'annotator',
    })
    data = data[(data['sentiment'].notnull()) & (
        data['attribute'].notnull()) & (data['entity'].notnull())]

    # map to coarse entities and attributes
    data.loc[:, 'entity'] = data.loc[:, 'entity'].apply(map_to_coarse_entities)
    data.loc[:, 'attribute'] = data.loc[:, 'attribute'].apply(
        map_to_coarse_attributes)

    # one hot encode target, default is sentiment
    if not predict_coarse_attributes_task:
        data = data.rename(columns={'sentiment': 'label'})
        data.loc[:, 'label'] = data.loc[:, 'label'].apply(
            one_hot_encode_ratings)
    else:
        data = data.rename(columns={'attribute': 'label'})
        data.loc[:, 'label'] = data.loc[:, 'label'].apply(
            one_hot_encode_coarse_attributes)

    # filter for entity
    data = data[data['entity'] == entity_filter]

    # embed text
    data['embedding'] = None
    for index, row in data.iterrows():
        data.at[index, 'embedding'] = text_processor(row['text'])

    data['split'] = split

    data['pseudo_labels'] = None

    return data


def one_hot_encode_ratings(rating):
    # set to None if label is to be excluded
    one_hot_sentiment_mapping = {
        'n': 0,
        '0': 1,
        'p': 2,
    }

    if rating not in one_hot_sentiment_mapping.keys():
        print(f'Rating not in map: {rating}')
    return one_hot_sentiment_mapping[rating]


def one_hot_encode_coarse_attributes(coarse_attribute):
    one_hot_coarse_attributes = {
        'general': 0,
        'price': 1,
        'experienced quality': 2,
        'safety and healthiness': 3,
        'trustworthy sources': 4,
        'environment': 5,
    }
    if coarse_attribute not in one_hot_coarse_attributes.keys():
        print(f'Attribute not in map: {coarse_attribute}')
    return one_hot_coarse_attributes[coarse_attribute]


def map_to_coarse_entities(entity):
    od_coarse_entities = {
        'g': 'organic',
        'p': 'organic',
        'f': 'organic',
        'c': 'organic',

        'cg': 'conventional',
        'cp': 'conventional',
        'cf': 'conventional',
        'cc': 'conventional',

        'gg': 'GMO'
    }
    if entity not in od_coarse_entities.keys():
        print(f'Entity not in map: {entity}')
    return od_coarse_entities[entity]


def map_to_coarse_attributes(attribute):
    od_coarse_attributes = {
        'g': 'general',
        'p': 'price',

        't': 'experienced quality',
        'q': 'experienced quality',

        's': 'safety and healthiness',
        'h': 'safety and healthiness',
        'c': 'safety and healthiness',

        'll': 'trustworthy sources',
        'or': 'trustworthy sources',
        'l': 'trustworthy sources',
        'av': 'trustworthy sources',

        'e': 'environment',
        'a': 'environment',
        'pp': 'environment',
    }
    if attribute not in od_coarse_attributes.keys():
        print(f'Attribute not in map: {attribute}')
    return od_coarse_attributes[attribute]


class OrganicDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)

        root = f'{self.root_data}organic/annotated_3rd_round/processed/train_test_validation V0.3'
        path_train = f'{root}/train/dataframe.csv'
        path_validation = f'{root}/validation/dataframe.csv'
        path_test = f'{root}/test/dataframe.csv'

        self.predict_coarse_attributes_task = args.get(
            'predict_coarse_attributes_task', False)
        self.entity_filter = args.get('entity_filter', 'organic')

        data_train = file_processor(path_train, self.text_processor, 'train',
                                    predict_coarse_attributes_task=self.predict_coarse_attributes_task,
                                    entity_filter=self.entity_filter)
        data_validation = file_processor(path_validation, self.text_processor,
                                         'validation', predict_coarse_attributes_task=self.predict_coarse_attributes_task,
                                         entity_filter=self.entity_filter)
        data_test = file_processor(path_test, self.text_processor, 'test', sep=',',
                                   predict_coarse_attributes_task=self.predict_coarse_attributes_task,
                                   entity_filter=self.entity_filter)

        self.data = pd.concat([data_train, data_validation, data_test], ignore_index=True)

        self.annotators = self.data.annotator.unique().tolist()
        self.data = self.data.to_dict('records')

        no_shuffle = args.get('no_shuffle', False)
        if no_shuffle is False:
            self.data_shuffle(split_included=True)

        self.pseudo_labels_key = 'pseudo_labels'
