import sys
import math
import torch
import pickle
import pandas as pd
import subprocess
from collections import Counter
import matplotlib.pyplot as plt

import models.dawid_skene as ds
from solver import Solver
from training import training_loop
from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from utils import *

METHOD = 'mace'
TRAINING = True
MODES = ['validation', 'test', 'train']  # set to only ['train'] when training
# MODES = ['train']

LOCAL_FOLDER = 'train_02_14/sgd/nll'
DEVICE = torch.device('cuda')
MACE_PATH = '../../MACE'
MACE_ITER = 1000
DS_ARGS = {
    'algorithm': 'FDS',
    'verbose': True
}
if METHOD not in ['dawid_skene', 'mace', 'majority_voting']:
    sys.exit()

EPOCHS = 300
USE_EPOCH_FACTOR = True
NUM_DRAWS = 20

# [10, 100, 300], [10, 100, 200], [10, 100, 200]]
SAVE_MODEL_AT = [10, 100, 200, 500, 1000, 2000]
EARLY_STOPPING_INTERVAL = 10
EARLY_STOPPING_MARGIN = 1e-5
# MODEL_WEIGHTS_PATH = '../models/train_10_17/tripadvisor/pretraining_softmax/' + \
#     '0.88136_batch64_lr0.00031867445707134466_20201019-092128_epoch300.pt'

USE_SOFTMAX = True
# macro doesn't make too much sense since we're calculating f1 after every batch --> micro
AVERAGING_METHOD = 'micro'
LR_INT = [1e-6, 1e-3]
BATCH_SIZES = [64]
DEEP_RANDOMIZATION = True
OPTIMIZER = 'sgd'


# # #  Setup  # # #
# # Parameters dependent on dataset # #

# label_dim = 2
# annotator_dim = 2
# loss = 'nll'
# dataset_name = 'wikipedia'
# task = 'toxicity'
# percentage = 0.05
# group_by_gender = True
# only_male_female = True
# dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender,
#                            percentage=percentage, only_male_female=only_male_female)

# label_dim = 3
# annotator_dim = 10
# loss = 'nll'
# padding_length = 136
# predict_coarse_attributes_task = False
# dataset_name = 'organic'
# domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
# dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
#                          padding_length=padding_length, domain_embedding_path=domain_embedding_path)
# task = 'sentiment'
# if predict_coarse_attributes_task:
#     task = 'coarse_attributes'
# if domain_embedding_path is not '':
#     task += '_fine_tuned_emb'
# epoch_factor = 10

label_dim = 3
annotator_dim = 38
loss = 'nll'
dataset_name = 'emotion'
domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
dataset = EmotionDataset(device=DEVICE, domain_embedding_path=domain_embedding_path)
emotion = 'valence'
dataset.set_emotion(emotion)
task = emotion
if domain_embedding_path is not '':
    task += '_fine_tuned_emb'
epoch_factor = 10

# label_dim = 2
# annotator_dim = 2
# loss = 'nll'
# one_dataset_one_annotator = False
# dataset = TripAdvisorDataset(device=DEVICE, one_dataset_one_annotator=one_dataset_one_annotator)
# dataset_name = 'tripadvisor'
# task = 'gender'
# if one_dataset_one_annotator:
#     task = 'dataset'
# epoch_factor = 2

local_folder = f'{LOCAL_FOLDER}/{METHOD}/{dataset_name}/{task}'

for mode in MODES:
    dataset.set_mode(mode)

    labels_path_root = f"../data/{METHOD}/{dataset_name}"
    if dataset_name is 'tripadvisor':
        if one_dataset_one_annotator:
            labels_path_root += '/1.3'
        else:
            labels_path_root += '/1.2'
    labels_path = labels_path_root
    labels_path += f'/sample_label_map_{dataset.mode}.pkl'

    solver_params = {
        'device': DEVICE,
        'label_dim': label_dim,
        'annotator_dim': annotator_dim,
        'averaging_method': AVERAGING_METHOD,
        'use_softmax': USE_SOFTMAX,
        'loss': loss,
        'optimizer_name': OPTIMIZER,
        'early_stopping_margin': EARLY_STOPPING_MARGIN,
        'save_at': SAVE_MODEL_AT,
    }
    if dataset_name is 'tripadvisor' or dataset_name is 'organic':
        pseudo_root = f'../models/{LOCAL_FOLDER}/{dataset_name}/{task}'
        pseudo_func_args = {
            'pseudo_root': pseudo_root,
            'phase': 'individual_training',
        }
        pseudo_model_path_func = get_pseudo_model_path

        solver_params.update({
            'pseudo_annotators': dataset.annotators,
            'pseudo_model_path_func': pseudo_model_path_func,
            'pseudo_func_args': pseudo_func_args,
        })
        # solver needed only for pseudo labels
        solver = Solver(dataset, 1e-5, BATCH_SIZES[0],
                        **solver_params)

    # dataset.set_mode('test')
    # print(set([point['text'] for point in dataset]))
    # sys.exit()

    data = {}
    for data_point in dataset:
        annotator = data_point['annotator']
        text = data_point['text']
        annotation = data_point['label'].item()

        if text not in data:
            data[text] = {}
        data[text][annotator] = [annotation]
        for pseudo_ann in list(data_point['pseudo_labels'].keys()):
            data[text][pseudo_ann] = [data_point['pseudo_labels'][pseudo_ann].item()]

    if METHOD == 'dawid_skene':
        # dawid_skene only for train set
        samples, labels = ds.run(data, DS_ARGS)

        # save labels in pickle
        labels_map = {samples[i]: labels[i] for i in range(len(samples))}
        f = open(labels_path, "wb")
        pickle.dump(labels_map, f)
        f.close()
        dataset.remove_pseudo_labels()

    if METHOD == 'mace':
        mace_data = [[ann_item[1][0] for ann_item in list(data_item[1].items())] for data_item in list(data.items())]
        mace_df = pd.DataFrame(mace_data)
        mace_csv_path = f'{labels_path_root}/crowdsourced_labels.csv'
        mace_df.to_csv(mace_csv_path, header=False, index=False)

        # run MACE with bash command
        %cd $labels_path_root
        src_path = '../../..'
        if dataset_name is 'tripadvisor':
            src_path += '/..'
        src_path += '/src'
        command = f'{src_path}/{MACE_PATH}/MACE --prefix mace_labels --iterations {MACE_ITER} {src_path}/{mace_csv_path}'
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)
        %cd $src_path

        # read labels
        mace_labels_path = f'{labels_path_root}/mace_labels.prediction'
        mace_labels = pd.read_csv(mace_labels_path, header=None)
        labels = mace_labels[0].tolist()

        # save labels in pickle
        labels_map = {data_item: labels[i] for i, data_item in enumerate(data)}
        f = open(labels_path, "wb")
        pickle.dump(labels_map, f)
        f.close()
        dataset.remove_pseudo_labels()

    if METHOD == 'majority_voting':
        mv_labels = {data_item: Counter([label[0] for label in list(data[data_item].values())]).most_common(1)[0][0] for data_item in data}
        f = open(labels_path, "wb")
        pickle.dump(mv_labels, f)
        f.close()
        dataset.remove_pseudo_labels()

    if TRAINING and mode == 'train':
        sample_label_map = {}
        with open(labels_path, 'rb') as f:
            sample_label_map = pickle.load(f)
        dataset.use_custom_labels(sample_label_map)
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS)
        epochs = EPOCHS
        if USE_EPOCH_FACTOR:
            epochs = EPOCHS * epoch_factor
        fit_params = {
            'return_f1': True,
            'deep_randomization': DEEP_RANDOMIZATION,
            'early_stopping_interval': EARLY_STOPPING_INTERVAL,
            'epochs': epochs,
            'pretrained_basic': False,
            'basic_only': True,
        }
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, epochs,
                    solver_params, fit_params)
