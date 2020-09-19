import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.tripadvisor import TripAdvisorDataset
from datasets import collate_wrapper
from models.ipa2lt_head import Ipa2ltHead


class Solver(object):

    def __init__(self, dataset, learning_rate, batch_size, momentum=0.9, model_weights_path='',
                 writer=None, device=torch.device('cpu'), verbose=True,
                 embedding_dim=50, label_dim=5, annotator_dim=2, pretrained_model=None,
                 ):
        self.learning_rate = learning_rate
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
        self.pretrained_model = pretrained_model

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
        if self.pretrained_model is None:
            model = self._get_model()
        else:
            model = self.pretrained_model

        if self.label_dim is 2:
            criterion = nn.BCELoss()
        elif self.label_dim > 2:
            criterion = nn.CrossEntropyLoss()
        optimizers = [optim.AdamW([
            {'params': model.basic_network.parameters()},
            {'params': model.bias_matrices[i].parameters()},
        ],
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
            for i in range(self.annotator_dim):
                # switch to current annotator
                annotator = self.dataset.annotators[i]
                self.dataset.set_annotator_filter(annotator)
                optimizer = optimizers[i]

                # training
                self.dataset.set_mode('train')
                train_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, collate_fn=collate_wrapper)
                self.fit_epoch(model, optimizer, criterion,
                               train_loader, annotator, i, epoch, loss_history)

                # validation
                self.dataset.set_mode('validation')
                val_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, collate_fn=collate_wrapper)
                if return_f1:
                    _, _, f1 = self.fit_epoch(model, optimizer, criterion,
                                              val_loader, annotator, i, epoch, loss_history, mode='train', return_metrics=True)
                else:
                    self.fit_epoch(model, optimizer, criterion,
                                   val_loader, annotator, i, epoch, loss_history, mode='validation')

        self._print('Finished Training' + 20 * ' ')
        self._print('sum of first 10 losses: ', sum(loss_history[0:10]))
        self._print('sum of last  10 losses: ', sum(loss_history[-10:]))

        if return_f1:
            return model, f1

        return model

    def fit_epoch(self, model, opt, criterion, data_loader, annotator, annotator_idx, epoch, loss_history, mode='train', return_metrics=False):
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
                f'Annotator {annotator} - Epoch {epoch}: Step {i} / {len_data_loader}' + 10 * ' ', end='\r')
            inputs, labels = data.input, data.target
            opt.zero_grad()

            # Generate predictions
            if annotator_idx is not None:
                outputs = model(inputs)[annotator_idx]
            else:
                outputs = model(inputs)

            # Compute Loss:
            loss = criterion(outputs.float(), labels)

            # performance measures of the batch
            predictions = outputs.argmax(dim=1)
            accuracy, precision, recall, f1 = self.performance_measures(predictions, labels)

            # statistics for logging
            current_batch_size = inputs.shape[0]
            divisor = (i - 1) * self.batch_size + current_batch_size
            mean_loss = ((i - 1) * self.batch_size * mean_loss +
                         loss.item() * current_batch_size) / divisor
            mean_accuracy = (mean_accuracy * self.batch_size * (i - 1) + accuracy.item() * current_batch_size) / divisor
            mean_precision = (mean_precision * self.batch_size * (i - 1) + precision.item() * current_batch_size) / divisor
            mean_recall = (mean_recall * self.batch_size * (i - 1) + recall.item() * current_batch_size) / divisor
            mean_f1 = (mean_f1 * self.batch_size * (i - 1) + f1.item() * current_batch_size) / divisor
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
    def performance_measures(predictions, labels):
        if predictions.device.type == 'cuda' or labels.device.type == 'cuda':
            predictions, labels = predictions.cpu(), labels.cpu()

        # averaging for multiclass targets, can be one of [‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        accuracy = accuracy_score(labels, predictions)
        average = 'weighted'
        zero_division = 0
        precision = precision_score(labels, predictions, average=average, zero_division=zero_division)
        recall = recall_score(labels, predictions, average=average, zero_division=zero_division)
        f1 = f1_score(labels, predictions, average=average, zero_division=zero_division)

        return accuracy, precision, recall, f1
