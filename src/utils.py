import os
import numpy as np
from scipy.special import exp10
from torch.utils.tensorboard import SummaryWriter


def get_writer(path, stem, current_time, params):
    if stem is '':
        stem = 'writer'
    log_dir = path + stem
    for key, value in params.items():
        log_dir += f'_{key}{value}'
    log_dir += f'_{current_time}'
    return SummaryWriter(log_dir=log_dir)


def get_model_path(path, stem, current_time, params, f1=0.0):
    first = True
    if stem is not '':
        path += f'{stem}'
        first = False
    if f1 is not 0.0:
        if not first:
            path += '_'
        path += '{:1.5f}'.format(f1)
        first = False
    for key, value in params.items():
        if not first:
            path += '_'
        path += f'{key}{value}'
        first = False
    path += f'_{current_time}'
    return path


def get_pseudo_model_path(pseudo_root, annotator, phase=''):
    """
    This function has to have an annotator argument,
    all other arguments should be provided to the solver separately.

    It assumes all model paths start with the f1 score and use '_' as a separator!
    """
    root = f'{pseudo_root}/'
    if phase is not '':
        root += f'{phase}/'
    root += f'{annotator}'
    f1s = []
    for model_path in os.listdir(root):
        f1, rest = (lambda x: (x[0], x[1:]))(model_path.split('_'))
        f1s.append(float(f1))
    f1s = np.asarray(f1s)
    idx = f1s.argmax()
    return f'{root}/{os.listdir(root)[idx]}'


def get_pseudo_model_path_tripadvisor(pseudo_root, annotator, phase=''):
    """
    This function has to have an annotator argument,
    all other arguments should be provided to the solver separately.

    It assumes all model paths start with the f1 score and use '_' as a separator!
    """
    path_dict = {
        'f': 'female/ipa_0.89038_batch64_lr0.0003306989309627488_20200924-222149.pt',
        'm': 'male/ipa_0.89060_batch64_lr8.299248182022548e-05_20200925-075432.pt',
    }
    return f'{pseudo_root}/{path_dict[annotator]}'


def get_best_model_path(path_to_models):
    """
    This function has to have an annotator argument,
    all other arguments should be provided to the solver separately.

    It assumes all model paths start with the f1 score and use '_' as a separator!
    """
    f1s = []
    for model_path in os.listdir(path_to_models):
        f1, rest = (lambda x: (x[0], x[1:]))(model_path.split('_'))
        f1s.append(float(f1))
    f1s = np.asarray(f1s)
    idx = f1s.argmax()
    return f'{path_to_models}/{os.listdir(path_to_models)[idx]}'


def get_learning_rates(start, end, num_draws):
    return exp10(-np.random.uniform(-np.log10(start), -np.log10(end), size=num_draws))
