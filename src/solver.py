import copy
import datetime
import time
import pytz
import itertools
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from datasets.tripadvisor import TripAdvisorDataset
from datasets import collate_wrapper
from models.ipa2lt_head import Ipa2ltHead


class Solver(object):

    def __init__(self, dataset, learning_rate, batch_size, momentum=0.9, model_weights_path='',
                 writer=None, device=torch.device('cpu'), verbose=True,
                 embedding_dim=50, label_dim=2, annotator_dim=2,
                 ):
        self.learning_rate = learning_rte
        self.batch_size = batch_size
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.label_dim = label_dim
        self.annotator_dim = annotator_dim
        self.momentum = momentum
        self.model_weights_path = model_weights_path
        self.device = device
        self.writer = writer
        self.verbose = verbose

    def _get_model(self):
        model = Ipa2ltHead(self.embedding_dim, self.label_dim, self.annotator_dim)
        if self.model_weights_path is not '':
            print(
                f'Training model with weights of file {self.model_weights_path}')
            model.load_state_dict(torch.load(self.model_weights_path))
        model.to(self.device)

        return model

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def fit(self, epochs, return_f1=False):
        model = self._get_model()

        if self.label_dim is 2:
            criterion = nn.BCELoss()
        elif self.label_dim > 2:
            criterion = nn.CrossEntropyLoss()
        optimizers = [optim.AdamW([model.basic_network.parameters(), model.bias_matrices[i].parameters()],
                                  lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                      for i in range(self.annotator_dim)]

        loss_history = []
        inputs = 0
        labels = 0
        f1 = 0.0

        # self._print('START TRAINING')
        self._print(
            f'learning rate: {self.learning_rate} - batch size: {self.batch_size}')
        for epoch in range(epochs):
            # TODO - loop over all annotators for training and do evaluation differently (maybe with latent truth??)
            # training
            self.dataset.set_mode('train')
            train_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, collate_fn=collate_wrapper)
            self.fit_epoch(model, optimizer, criterion,
                           train_loader, epoch, loss_history)

            # validation
            self.dataset.set_mode('validation')
            val_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, collate_fn=collate_wrapper)
            if return_f1:
                _, _, f1 = self.fit_epoch(model, optimizer, criterion,
                                          val_loader, epoch, loss_history, mode='train', return_metrics=True)
            else:
                self.fit_epoch(model, optimizer, criterion,
                               val_loader, epoch, loss_history, mode='validation')

        self._print('Finished Training' + 20 * ' ')
        self._print('sum of first 10 losses: ', sum(loss_history[0:10]))
        self._print('sum of last  10 losses: ', sum(loss_history[-10:]))

        if return_f1:
            return model, f1

        return model

    def fit_epoch(self, model, opt, criterion, data_loader, annotator, epoch, loss_history, mode='train', return_metrics=False):
        mean_loss = 0.0
        mean_accuracy = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
        mean_f1 = 0.0
        len_data_loader = len(data_loader)
        for i, data in enumerate(data_loader, 1):
            # Prepare inputs to be passed to the model
            # Prepare labels for the Loss computation
            self._print(
                f'Epoch {epoch}: Step {i} / {len_data_loader}', end='\r')
            inputs, labels = data.input, data.target
            opt.zero_grad()

            # Generate predictions
            outputs = model(inputs).squeeze(1)

            # Compute Loss:
            loss = criterion(outputs.float(), labels.float())

            # performance measures of the batch
            accuracy, precision, recall, f1 = self.performance_measures(outputs, labels, self.hinge_loss)

            # statistics for logging
            current_batch_size = inputs.shape[0]
            divisor = (i - 1) * batch_size + current_batch_size
            mean_loss = ((i - 1) * batch_size * mean_loss +
                         loss.item() * current_batch_size) / divisor
            mean_accuracy = (mean_accuracy * batch_size * (i - 1) + accuracy.item() * current_batch_size) / divisor
            mean_precision = (mean_precision * batch_size * (i - 1) + precision.item() * current_batch_size) / divisor
            mean_recall = (mean_recall * batch_size * (i - 1) + recall.item() * current_batch_size) / divisor
            mean_f1 = (mean_f1 * batch_size * (i - 1) + f1.item() * current_batch_size) / divisor
            loss_history.append(loss.item())

            if mode is 'train':
                # Update gradients
                loss.backward()

                # Optimization step
                opt.step()

            if self.writer is not None:
                self.writer.add_scalar(f'Loss/Annotator {annotator}/{mode}', mean_loss, epoch)
                self.writer.add_scalar(
                    f'Accuracy/Annotator {annotator}/{mode}', mean_accuracy, epoch)
                self.writer.add_scalar(
                    f'Precision/Annotator {annotator}/{mode}', mean_precision, epoch)
                self.writer.add_scalar(f'Recall/Annotator {annotator}/{mode}', mean_recall, epoch)
                self.writer.add_scalar(f'F1 score/Annotator {annotator}/{mode}', mean_f1, epoch)

            if return_metrics:
                return mean_loss, mean_accuracy, mean_f1

    @staticmethod
    def performance_measures(outputs, labels, hinge_loss=False):
        if hinge_loss:
            preds_mask = outputs > 0.0
            labels_mask = labels > 0.0
        else:
            preds_mask = outputs > 0.5
            labels_mask = labels > 0.5
        true_positives = torch.ones_like(
            labels)[preds_mask & labels_mask].sum()
        true_negatives = torch.ones_like(
            labels)[(preds_mask == False) & (labels_mask == False)].sum()
        false_positives = torch.ones_like(
            labels)[preds_mask & (labels_mask == False)].sum()
        false_negatives = torch.ones_like(
            labels)[(preds_mask == False) & labels_mask].sum()
        batch_size = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / batch_size
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1
