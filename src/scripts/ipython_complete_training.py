import os
import torch
import numpy as np
import pytz
import datetime
from scipy.special import exp10
from itertools import product

from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from training import training_loop
from utils import *

# Config
# EPOCHS_PHASES = [10, 30, 30, 30]
# NUM_DRAWS_PHASES = [2, 3, 3, 3]
EPOCHS_PHASES = [100, 200, 300, 300, 300, 300]
NUM_DRAWS_PHASES = [10, 20, 20, 20, 20, 20]
# [10, 100, 300], [10, 100, 200], [10, 100, 200]]
SAVE_MODEL_AT_PHASES = [[10], [10, 100, 200, 500, 1000, 2000], [10, 100, 200, 500, 1000, 2000],
                        [10, 100, 200, 500, 1000, 2000], [10, 100, 200, 500, 1000, 2000], [10, 100, 200, 500, 1000, 2000]]
EARLY_STOPPING_INTERVAL = 10
EARLY_STOPPING_MARGIN = 1e-5
LOCAL_FOLDER = 'train_02_14/sgd/nll'
# MODEL_WEIGHTS_PATH = '../models/train_10_17/tripadvisor/pretraining_softmax/' + \
#     '0.88136_batch64_lr0.00031867445707134466_20201019-092128_epoch300.pt'

STEM = ''
USE_SOFTMAX = True
# macro doesn't make too much sense since we're calculating f1 after every batch --> micro
AVERAGING_METHOD = 'micro'
LR_INT = [1e-6, 1e-3]
BATCH_SIZES = [64]
DEVICE = torch.device('cuda')
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

# label_dim = 3
# annotator_dim = 38
# loss = 'nll'
# dataset_name = 'emotion'
# domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
# dataset = EmotionDataset(device=DEVICE, domain_embedding_path=domain_embedding_path)
# emotion = 'valence'
# dataset.set_emotion(emotion)
# task = emotion
# if domain_embedding_path is not '':
#     task += '_fine_tuned_emb'

label_dim = 2
annotator_dim = 2
loss = 'nll'
one_dataset_one_annotator = False
dataset = TripAdvisorDataset(device=DEVICE, one_dataset_one_annotator=one_dataset_one_annotator)
dataset_name = 'tripadvisor'
task = 'gender'

local_folder = f'{LOCAL_FOLDER}/{dataset_name}/{task}'


# # Parameters independent of dataset # #
solver_params = {
    'device': DEVICE,
    'label_dim': label_dim,
    'annotator_dim': annotator_dim,
    'averaging_method': AVERAGING_METHOD,
    'use_softmax': USE_SOFTMAX,
    'loss': loss,
    'optimizer_name': OPTIMIZER,
    'early_stopping_margin': EARLY_STOPPING_MARGIN,
}
fit_params = {
    'return_f1': True,
    'deep_randomization': DEEP_RANDOMIZATION,
    'early_stopping_interval': EARLY_STOPPING_INTERVAL,
}
models_root_path = f'../models/{local_folder}'
pseudo_func_args = {
    'pseudo_root': models_root_path,
    'phase': 'individual_training',
}
pseudo_model_path_func = get_pseudo_model_path

# # #  Training  # # #
# Emotions Loop (comment out as needed)
# for emotion in dataset.emotions:

# Full training loop (comment out as needed)
# phases = ['individual_training', 'pretraining',
#           'ltnet_true', 'ltnet_pseudo',
#           'basic_true', 'basic_pseudo']
phases = ['ltnet_true']
for phase in phases:
    print(f'NEW PHASE - now in phase {phase}')
    if phase is 'individual_training':
        # Annotator Loop (comment out as needed)
        for annotator in dataset.annotators:
            learning_rates = get_learning_rates(
                LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[0])
            solver_params_copy = solver_params.copy()
            solver_params_copy.update({
                'save_at': SAVE_MODEL_AT_PHASES[0],
            })
            fit_params_copy = fit_params.copy()
            fit_params_copy.update({
                'epochs': EPOCHS_PHASES[0],
                'basic_only': True,
                'single_annotator': annotator,
            })
            training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[0],
                          solver_params_copy, fit_params_copy, phase_path=phase, annotator_path=annotator)

    if phase is 'pretraining':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[1])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[1],
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[1],
            'basic_only': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[1],
                      solver_params_copy, fit_params_copy, phase_path=phase)

    if phase is 'ltnet_true':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[2])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[2],
            # get best model from pretraining
            # 'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[2],
            'pretrained_basic': False,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[3],
                      solver_params_copy, fit_params_copy, phase_path=phase)

    if phase is 'ltnet_pseudo':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[3])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[3],
            # get best model from pretraining
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),
            'pseudo_annotators': dataset.annotators,
            'pseudo_model_path_func': pseudo_model_path_func,
            'pseudo_func_args': pseudo_func_args,
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[3],
            'pretrained_basic': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[2],
                      solver_params_copy, fit_params_copy, phase_path=phase)
        dataset.remove_pseudo_labels()

    if phase is 'basic_true':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[4])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[4],
            # get best model from pretraining
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[4],
            'pretrained_basic': True,
            'basic_only': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[5],
                      solver_params_copy, fit_params_copy, phase_path=phase)

    if phase is 'basic_pseudo':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[5])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[5],
            # get best model from pretraining
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),
            'pseudo_annotators': dataset.annotators,
            'pseudo_model_path_func': pseudo_model_path_func,
            'pseudo_func_args': pseudo_func_args,
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[5],
            'pretrained_basic': True,
            'basic_only': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[2],
                      solver_params_copy, fit_params_copy, phase_path=phase)
        dataset.remove_pseudo_labels()
