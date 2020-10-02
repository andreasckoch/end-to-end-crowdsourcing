import os
import torch
import numpy as np
import pytz
import datetime
from scipy.special import exp10
from itertools import product

from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset
from utils import get_writer, get_model_path

# Config
EPOCHS = 1000
SAVE_MODEL_AT = [10, 50, 100, 500]
LOCAL_FOLDER = 'train_10_02/full_training_mediocre_base'
MODEL_WEIGHTS_PATH = '../models/train_09_29/base_network_pretraining/ipa_0.82707_batch64_lr0.000819819798011974_20200929-184901_epoch10.pt'
STEM = 'ipa'
LR_INT = [1e-7, 1e-3]
NUM_DRAWS = 20
BATCH_SIZES = [64]
DEVICE = torch.device('cuda')

# Setup
learning_rates = exp10(-np.random.uniform(-np.log10(LR_INT[0]), -np.log10(LR_INT[1]), size=NUM_DRAWS))
dataset = TripAdvisorDataset(device=DEVICE)

# Training Loop
for batch_size, lr in product(BATCH_SIZES, learning_rates):
    # For Documentation
    current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
    hyperparams = {'batch': batch_size, 'lr': lr}
    writer = get_writer(path=f'../logs/{LOCAL_FOLDER}/', stem=STEM,
                        current_time=current_time, params=hyperparams)

    # Save model path
    if LOCAL_FOLDER != '' and not os.path.exists('../models/' + LOCAL_FOLDER + '/'):
        os.makedirs('../models/' + LOCAL_FOLDER + '/')
    path = '../models/'
    if LOCAL_FOLDER != '':
        path += LOCAL_FOLDER + '/'
    save_params = {'stem': STEM, 'current_time': current_time, 'hyperparams': hyperparams}

    # Training
    solver = Solver(dataset, lr, batch_size, writer=writer, device=DEVICE, model_weights_path=MODEL_WEIGHTS_PATH,
                    save_path_head=path, save_at=SAVE_MODEL_AT, save_params=save_params)
    model, f1 = solver.fit(epochs=EPOCHS, return_f1=True, pretrained_basic=True)

    # Save model
    model_path = get_model_path(path, STEM, current_time, hyperparams, f1)
    torch.save(model.state_dict(), path + f'_epoch{EPOCHS}.pt')
