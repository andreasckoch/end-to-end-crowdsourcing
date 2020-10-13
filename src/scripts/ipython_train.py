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
from utils import get_writer, get_model_path

# Config
EPOCHS = 1000
SAVE_MODEL_AT = []
LOCAL_FOLDER = 'train_10_12/individual_classifiers'
MODEL_WEIGHTS_PATH = ''
PSEUDO_LABELS_MODEL_PATHS = {
    # 'f': {'pseudo_annotator': 'm',
    #       'model_path': '../models/train_09_24/male/ipa_0.89060_batch64_lr8.299248182022548e-05_20200925-075432.pt'},
    # 'm': {'pseudo_annotator': 'f',
    #       'model_path': '../models/train_09_24/female/ipa_0.89038_batch64_lr0.0003306989309627488_20200924-222149.pt'}
}
STEM = 'ipa'
LABEL_DIM = 3
LR_INT = [1e-6, 1e-3]
NUM_DRAWS = 20
BATCH_SIZES = [64]
DEVICE = torch.device('cuda')
# emotion = 'valence'

# Setup
learning_rates = exp10(-np.random.uniform(-np.log10(LR_INT[0]), -np.log10(LR_INT[1]), size=NUM_DRAWS))
dataset = EmotionDataset(device=DEVICE)
# dataset = TripAdvisorDataset(device=DEVICE)

# Emotions Loop (comment out as needed)
for emotion in dataset.emotions:
    dataset.set_emotion(emotion)

    # Annotator Loop (comment out as needed)
    for annotator in dataset.annotators:

        # Training Loop
        for batch_size, lr in product(BATCH_SIZES, learning_rates):
            # sub path
            sub_path = f'{LOCAL_FOLDER}/{emotion}/{annotator}/'

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
                            pseudo_labels_model_paths=PSEUDO_LABELS_MODEL_PATHS)
            model, f1 = solver.fit(epochs=EPOCHS, return_f1=True, basic_only=True, single_annotator=annotator)

            # Save model
            model_path = get_model_path(path, STEM, current_time, hyperparams, f1)
            torch.save(model.state_dict(), model_path + f'_epoch{EPOCHS}.pt')
