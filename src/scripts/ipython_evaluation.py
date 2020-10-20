import torch
import os

from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset

# Setup
LABEL_DIM = 3
LABELS = ['neg', 'neutral', 'pos']
ANNOTATOR_DIM = 38
DEVICE = torch.device('cuda')
USE_SOFTMAX = True
MODEL_ROOT = '../models'
LOGS_ROOT = '../logs'
LOCAL_FOLDER = 'train_10_19/complete_training_test/anger/full_training'
# version = 3
# model_weights_path = '' '../models/train_10_17/tripadvisor/full_training_good_model_fix_base_softmax/' + \
#     '0.87622_batch64_lr1.5097526416899163e-05_20201019-093149_epoch300.pt'
# output_file_path = f'../logs/train_10_17/tripadvisor/full_training_good_model_fix_base_softmax/evaluation_sample{version}.txt'

dataset = EmotionDataset(device=DEVICE)
dataset.set_emotion('anger')
# dataset = TripAdvisorDataset(device=DEVICE)
model_root = f'{MODEL_ROOT}/{LOCAL_FOLDER}'
log_root = f'{LOGS_ROOT}/{LOCAL_FOLDER}'
for model_path in os.listdir(model_root):
    if model_path.endswith('.pt'):
        model_full_path = f'{model_root}/{model_path}'
        hyperparams, _ = (lambda x: (x[:-1], x[-1]))(model_path.split('.pt'))
        log_full_path = f'{model_root}/{hyperparams[0]}'
        if USE_SOFTMAX:
            log_full_path += '_softmax'
        else:
            log_full_path += '_sigmoid'
        log_full_path += '.txt'
        solver = Solver(dataset, 1e-5, 32, model_weights_path=model_full_path, device=torch.device('cuda'),
                        annotator_dim=ANNOTATOR_DIM, label_dim=LABEL_DIM, use_softmax=USE_SOFTMAX)

        solver.evaluate_model(output_file_path=log_full_path, labels=LABELS)
