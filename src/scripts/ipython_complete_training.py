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
EPOCHS_PHASES = [100, 300, 300, 300]
NUM_DRAWS_PHASES = [5, 10, 10, 10]
# [10, 100, 300], [10, 100, 200], [10, 100, 200]]
SAVE_MODEL_AT_PHASES = [[], [10, 100, 200], [10, 100, 200], [10, 100, 200]]
LOCAL_FOLDER = 'train_11_09/complete_training'
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

# # #  Setup  # # #
# # Parameters dependent on dataset # #

# label_dim = 2
# annotator_dim = 2
# dataset_name = 'wikipedia'
# task = 'toxicity'
# percentage = 0.05
# group_by_gender = True
# only_male_female = True
# dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender,
#                            percentage=percentage, only_male_female=only_male_female)

# label_dim = 3
# annotator_dim = 10
# padding_length = 136
# predict_coarse_attributes_task = False
# dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
#                          padding_length=padding_length)
# dataset_name = 'organic'
# task = 'sentiment'
# if predict_coarse_attributes_task:
#     task = 'coarse_attributes'

# label_dim = 3
# annotator_dim = 38
# dataset = EmotionDataset(device=DEVICE)
# emotion = 'valence'
# dataset.set_emotion(emotion)
# dataset_name = 'emotion'
# task = emotion

label_dim = 2
annotator_dim = 2
dataset = TripAdvisorDataset(device=DEVICE)
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
}
fit_params = {
    'return_f1': True,
    'deep_randomization': DEEP_RANDOMIZATION,
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
phases = ['individual_training', 'pretraining',
          'full_training', 'no_pseudo_labeling']
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

    if phase is 'full_training':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[2])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[2],
            # get best model from pretraining
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),
            'pseudo_annotators': dataset.annotators,
            'pseudo_model_path_func': pseudo_model_path_func,
            'pseudo_func_args': pseudo_func_args,
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[2],
            'pretrained_basic': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[2],
                      solver_params_copy, fit_params_copy, phase_path=phase)

    if phase is 'no_pseudo_labeling':
        learning_rates = get_learning_rates(
            LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[3])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[3],
            # get best model from pretraining
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[3],
            'pretrained_basic': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[3],
                      solver_params_copy, fit_params_copy, phase_path=phase)
