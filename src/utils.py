from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


def get_writer(path, stem, current_time, params):
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


def get_pseudo_model_path(pseudo_root, emotion, annotator):
    """
    This function has to have an annotator argument,
    all other arguments should be provided to the solver separately.

    It assumes all model paths start with the f1 score and use '_' as a separator!
    """
    root = f'{pseudo_root}/{emotion}/{annotator}'
    f1s = []
    for model_path in os.listdir(root):
        f1, rest = (lambda x: (x[0], x[1:]))(model_path.split('_'))
        f1s.append(float(f1))
    f1s = np.asarray(f1s)
    idx = f1s.argmax()
    return f'{root}/{os.listdir(root)[idx]}'
