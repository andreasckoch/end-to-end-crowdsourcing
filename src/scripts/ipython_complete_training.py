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
from training import training_loop
from utils import *

# Config
EPOCHS_PHASES = [30, 50, 20, 20]
NUM_DRAWS_PHASES = [5, 10, 6, 6]
SAVE_MODEL_AT_PHASES = [[], [10, 30], [2, 10], [10]]  # [10, 100, 300], [10, 100, 200], [10, 100, 200]]
LOCAL_FOLDER = 'train_10_27/complete_training'
# MODEL_WEIGHTS_PATH = '../models/train_10_17/tripadvisor/pretraining_softmax/' + \
#     '0.88136_batch64_lr0.00031867445707134466_20201019-092128_epoch300.pt'

STEM = ''
LABEL_DIM = 2
ANNOTATOR_DIM = 3
USE_SOFTMAX = True
AVERAGING_METHOD = 'macro'
LR_INT = [1e-6, 1e-3]
BATCH_SIZES = [64]
DEVICE = torch.device('cuda')

# # #  Setup  # # #
# # Parameters dependent on dataset # #
task = 'toxicity'
group_by_gender = True
percentage = 0.05
dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender, percentage=percentage)

# dataset = EmotionDataset(device=DEVICE)
# emotion = 'valence'
# dataset.set_emotion(emotion)

# dataset = TripAdvisorDataset(device=DEVICE)

local_folder = f'{LOCAL_FOLDER}/wikipedia/{task}'  # {emotion}'


# # Parameters independent of dataset # #
solver_params = {
    'device': DEVICE,
    'label_dim': LABEL_DIM,
    'annotator_dim': ANNOTATOR_DIM,
    'averaging_method': AVERAGING_METHOD,
    'use_softmax': USE_SOFTMAX,
}
fit_params = {
    'return_f1': True,
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
phases = ['individual_training', 'pretraining', 'full_training', 'no_pseudo_labeling']
for phase in phases:
    print(f'NEW PHASE - now in phase {phase}')
    # if phase is 'individual_training':
    #     # Annotator Loop (comment out as needed)
    #     for annotator in dataset.annotators:
    #         learning_rates = get_learning_rates(LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[0])
    #         solver_params_copy = solver_params.copy()
    #         solver_params_copy.update({
    #             'save_at': SAVE_MODEL_AT_PHASES[0],
    #         })
    #         fit_params_copy = fit_params.copy()
    #         fit_params_copy.update({
    #             'epochs': EPOCHS_PHASES[0],
    #             'basic_only': True,
    #             'single_annotator': annotator,
    #         })
    #         training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[0],
    #                       solver_params_copy, fit_params_copy, phase_path=phase, annotator_path=annotator)

    # if phase is 'pretraining':
    #     learning_rates = get_learning_rates(LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[1])
    #     solver_params_copy = solver_params.copy()
    #     solver_params_copy.update({
    #         'save_at': SAVE_MODEL_AT_PHASES[1],
    #     })
    #     fit_params_copy = fit_params.copy()
    #     fit_params_copy.update({
    #         'epochs': EPOCHS_PHASES[1],
    #         'basic_only': True,
    #     })
    #     training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[1],
    #                   solver_params_copy, fit_params_copy, phase_path=phase)

    if phase is 'full_training':
        learning_rates = get_learning_rates(LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[2])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[2],
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),  # get best model from pretraining
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
        learning_rates = get_learning_rates(LR_INT[0], LR_INT[1], NUM_DRAWS_PHASES[3])
        solver_params_copy = solver_params.copy()
        solver_params_copy.update({
            'save_at': SAVE_MODEL_AT_PHASES[3],
            'model_weights_path': get_best_model_path(f'{models_root_path}/{phases[1]}'),  # get best model from pretraining
        })
        fit_params_copy = fit_params.copy()
        fit_params_copy.update({
            'epochs': EPOCHS_PHASES[3],
            'pretrained_basic': True,
        })
        training_loop(dataset, BATCH_SIZES, learning_rates, local_folder, EPOCHS_PHASES[3],
                      solver_params_copy, fit_params_copy, phase_path=phase)
