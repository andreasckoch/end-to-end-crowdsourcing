import os
import torch
import numpy as np
import pytz
import datetime
from scipy.special import exp10
from itertools import product

from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from utils import *

# Config
EPOCHS = 50
SAVE_MODEL_AT = [10]
LOCAL_FOLDER = 'train_11_07/complete_training_test'
PHASE = 'full_training'

STEM = ''
USE_SOFTMAX = True
AVERAGING_METHOD = 'micro'
LR_INT = [1e-6, 1e-3]
NUM_DRAWS = 3
BATCH_SIZES = [64]
DEVICE = torch.device('cuda')
DEEP_RANDOMIZATION = True

# Setup
learning_rates = exp10(-np.random.uniform(-np.log10(LR_INT[0]), -np.log10(LR_INT[1]), size=NUM_DRAWS))

# label_dim = 3
# annotator_dim = 38
# dataset = EmotionDataset(device=DEVICE)
# dataset_name = 'emotion'
# task = 'valence'

# label_dim = 2
# annotator_dim = 2
# dataset = TripAdvisorDataset(device=DEVICE)
# dataset_name = 'tripadvisor'
# task = 'gender'

# label_dim = 2
# annotator_dim = 2
# task = 'toxicity'
# percentage = 0.05
# group_by_gender = True
# only_male_female = True
# dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender,
#                            percentage=percentage, only_male_female=only_male_female)
# dataset_name = 'wikipedia'
# task = 'gender'

label_dim = 3
annotator_dim = 10
padding_length = 136
predict_coarse_attributes_task = False
dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
                         padding_length=padding_length)
dataset_name = 'organic'
task = 'sentiment'
if predict_coarse_attributes_task:
    task = 'coarse_attributes'

local_folder = f'{LOCAL_FOLDER}/{dataset_name}/{task}'

models_root_path = f'../models/{local_folder}'
pseudo_func_args = {
    'pseudo_root': models_root_path,
    'phase': 'individual_training',
}
pseudo_model_path_func = get_pseudo_model_path

# model_weights_path = '../models/train_11_01/complete_training/wikipedia/toxicity/pretraining/' + \
#     '0.55428_batch64_lr0.0007943720850343734_20201101-213326_epoch10.pt'
model_weights_path = get_best_model_path(f'{models_root_path}/pretraining')


# Emotions Loop (comment out as needed)
# for emotion in dataset.emotions:
# emotion = 'valence'
# dataset.set_emotion(emotion)

# # Annotator Loop (comment out as needed)
# for annotator in dataset.annotators:

# Training Loop
for batch_size, lr in product(BATCH_SIZES, learning_rates):
    # sub path
    sub_path = f'{local_folder}/{PHASE}/'

    # For Documentation
    current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
    hyperparams = {'batch': batch_size, 'lr': lr}
    writer = get_writer(path=f'../logs/{sub_path}', stem=STEM,
                        current_time=current_time, params=hyperparams)

    # Save model path
    if LOCAL_FOLDER != '' and not os.path.exists('../models/' + sub_path):
        os.makedirs('../models/' + sub_path)
    path = '../models/'
    if LOCAL_FOLDER != '':
        path += sub_path
    save_params = {'stem': STEM, 'current_time': current_time, 'hyperparams': hyperparams}

    # Training

    solver = Solver(dataset, lr, batch_size, writer=writer, device=DEVICE, model_weights_path=model_weights_path,
                    save_path_head=path, save_at=SAVE_MODEL_AT, save_params=save_params, label_dim=label_dim,
                    annotator_dim=annotator_dim, averaging_method=AVERAGING_METHOD, use_softmax=USE_SOFTMAX,
                    pseudo_annotators=dataset.annotators,
                    pseudo_model_path_func=get_pseudo_model_path,
                    pseudo_func_args=pseudo_func_args,
                    )
    model, f1 = solver.fit(epochs=EPOCHS, return_f1=True, pretrained_basic=True, deep_randomization=DEEP_RANDOMIZATION)

    # Save model
    model_path = get_model_path(path, STEM, current_time, hyperparams, f1)
    torch.save(model.state_dict(), model_path + f'_epoch{EPOCHS}.pt')
