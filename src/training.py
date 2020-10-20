import os
import torch
import numpy as np
import pytz
import datetime
from scipy.special import exp10
from itertools import product

from solver import Solver
from utils import get_writer, get_model_path


def training_loop(dataset, batch_sizes, learning_rates, local_folder, epochs, solver_params,
                  fit_params, stem='', root='../models', phase_path='', annotator_path=''):

    # Training Loop
    for batch_size, lr in product(batch_sizes, learning_rates):
        # sub path
        sub_path = f'{local_folder}/'
        if phase_path is not '':
            sub_path += f'{phase_path}/'
        if annotator_path is not '':
            sub_path += f'{annotator_path}/'

        # For Documentation
        current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
        hyperparams = {'batch': batch_size, 'lr': lr}
        writer = get_writer(path=f'../logs/{sub_path}', stem=stem,
                            current_time=current_time, params=hyperparams)

        # Save model path
        if local_folder != '' and not os.path.exists('../models/' + sub_path):
            os.makedirs('../models/' + sub_path)
        path = '../models/'
        if local_folder != '':
            path += sub_path
        save_params = {'stem': stem, 'current_time': current_time, 'hyperparams': hyperparams}

        # Training
        solver = Solver(dataset, lr, batch_size, writer=writer, save_path_head=path, save_params=save_params,
                        **solver_params)
        model, f1 = solver.fit(**fit_params)

        # Save model
        model_path = get_model_path(path, stem, current_time, hyperparams, f1)
        torch.save(model.state_dict(), model_path + f'_epoch{epochs}.pt')