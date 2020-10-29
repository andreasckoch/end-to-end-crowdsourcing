import torch
import os

from solver import Solver
from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset

# Setup
LABEL_DIM = 3
LABELS = ['neg', 'neutral', 'pos']
ANNOTATOR_DIM = 38
DEVICE = torch.device('cuda')
USE_SOFTMAX = True
MODES = ['train', 'test']
MODEL_ROOT = '../models'
LOGS_ROOT = '../logs'
local_folder_root = 'train_10_20/no_pseudo_labeling/valence'
local_folder = f'{local_folder_root}/full_training'
pretrained_model_path = f'{MODEL_ROOT}/{local_folder_root}/pretraining/' + \
    '0.62222_batch32_lr0.0009900373459039093_20201020-163838_epoch1000.pt'

dataset = EmotionDataset(device=DEVICE)
dataset.set_emotion('valence')

# dataset = TripAdvisorDataset(device=DEVICE)

# task = 'toxicity'
# group_by_gender = True
# percentage = 0.05
# dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender, percentage=percentage)


model_root = f'{MODEL_ROOT}/{local_folder}'
log_root = f'{LOGS_ROOT}/{local_folder}'
for model_path in os.listdir(model_root):

    # modes loop (comment out as needed)
    for mode in MODES:

        if model_path.endswith('.pt'):
            model_full_path = f'{model_root}/{model_path}'
            hyperparams, _ = (lambda x: (x[:-1], x[-1]))(model_path.split('.pt'))
            log_full_path = f'{model_root}/{hyperparams[0]}'
            if USE_SOFTMAX:
                log_full_path += '_softmax'
            else:
                log_full_path += '_sigmoid'
            log_full_path += f'_{mode}'
            log_full_path += '.txt'
            solver = Solver(dataset, 1e-5, 32, model_weights_path=model_full_path, device=torch.device('cuda'),
                            annotator_dim=ANNOTATOR_DIM, label_dim=LABEL_DIM, use_softmax=USE_SOFTMAX)

            solver.evaluate_model(output_file_path=log_full_path, labels=LABELS, mode=mode, pretrained_basic_path=pretrained_model_path)
