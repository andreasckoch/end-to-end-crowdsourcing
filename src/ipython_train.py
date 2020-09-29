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
EPOCHS = 15
LOCAL_FOLDER = 'test'
STEM = 'ipa'
LR_INT = [1e-7, 1e-3]
NUM_DRAWS = 20
BATCH_SIZES = [32]
DEVICE = torch.device('cuda')
path = '../logs/test/'

# Setup
learning_rates = exp10(-np.random.uniform(-np.log10(LR_INT[0]), -np.log10(LR_INT[1]), size=NUM_DRAWS))
dataset = TripAdvisorDataset(device=DEVICE)

# Training Loop
for batch_size, lr in product(BATCH_SIZES, learning_rates):
    # For Documentation
    current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
    hyperparams = {'batch': batch_size, 'lr': lr}
    writer = get_writer(path, stem=STEM, current_time=current_time, params=hyperparams)

    # Training
    solver = Solver(dataset, lr, batch_size, writer=writer, device=DEVICE)
    model, f1 = solver.fit(epochs=EPOCHS, return_f1=True)

    # Save model
    if LOCAL_FOLDER != '' and not os.path.exists('../models/' + LOCAL_FOLDER + '/'):
        os.makedirs('../models/' + LOCAL_FOLDER + '/')
    path = '../models/'
    if LOCAL_FOLDER != '':
        path += LOCAL_FOLDER + '/'
    path = get_model_path(path, STEM, current_time, hyperparams, f1)
    torch.save(model.state_dict(), path)
