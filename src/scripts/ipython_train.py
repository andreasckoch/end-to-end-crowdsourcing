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
from utils import get_writer, get_model_path, get_pseudo_model_path, get_pseudo_model_path_tripadvisor

# Config
EPOCHS = 300
SAVE_MODEL_AT = [10, 100, 200]
LOCAL_FOLDER = 'train_10_17/tripadvisor/full_training_good_model_fix_base_softmax'
MODEL_WEIGHTS_PATH = '../models/train_10_17/tripadvisor/pretraining_softmax/' + \
    '0.88136_batch64_lr0.00031867445707134466_20201019-092128_epoch300.pt'
PSEUDO_ROOT_PATH = '../models/train_09_24'

STEM = ''
LABEL_DIM = 2
ANNOTATOR_DIM = 2
USE_SOFTMAX = True
AVERAGING_METHOD = 'macro'
LR_INT = [1e-6, 1e-3]
NUM_DRAWS = 20
BATCH_SIZES = [64]
DEVICE = torch.device('cuda')
# emotion = 'valence'

# Setup
learning_rates = exp10(-np.random.uniform(-np.log10(LR_INT[0]), -np.log10(LR_INT[1]), size=NUM_DRAWS))
# dataset = EmotionDataset(device=DEVICE)
dataset = TripAdvisorDataset(device=DEVICE)

# Emotions Loop (comment out as needed)
# for emotion in dataset.emotions:
# emotion = 'anger'
# dataset.set_emotion(emotion)

# # Annotator Loop (comment out as needed)
# for annotator in dataset.annotators:

# Training Loop
for batch_size, lr in product(BATCH_SIZES, learning_rates):
    # sub path
    sub_path = f'{LOCAL_FOLDER}/'  # {annotator}/'  # {emotion}/'

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
    solver = Solver(dataset, lr, batch_size, writer=writer, device=DEVICE, model_weights_path=MODEL_WEIGHTS_PATH,
                    save_path_head=path, save_at=SAVE_MODEL_AT, save_params=save_params, label_dim=LABEL_DIM,
                    annotator_dim=ANNOTATOR_DIM, averaging_method=AVERAGING_METHOD, use_softmax=USE_SOFTMAX,
                    pseudo_annotators=dataset.annotators,
                    pseudo_model_path_func=get_pseudo_model_path_tripadvisor,
                    pseudo_func_args={'pseudo_root': PSEUDO_ROOT_PATH}  # 'emotion': emotion})
                    )
    model, f1 = solver.fit(epochs=EPOCHS, return_f1=True, pretrained_basic=True, fix_base=True)

    # Save model
    model_path = get_model_path(path, STEM, current_time, hyperparams, f1)
    torch.save(model.state_dict(), model_path + f'_epoch{EPOCHS}.pt')
