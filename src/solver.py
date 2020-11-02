import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import compress

from datasets.tripadvisor import TripAdvisorDataset
from models.ipa2lt_head import Ipa2ltHead
from models.basic import BasicNetwork
from utils import get_model_path


class Solver(object):

    def __init__(self, dataset, learning_rate, batch_size, momentum=0.9, model_weights_path='',
                 writer=None, device=torch.device('cpu'), verbose=True,
                 embedding_dim=50, label_dim=2, annotator_dim=2, averaging_method='macro',
                 save_path_head=None, save_at=None, save_params=None, use_softmax=True,
                 pseudo_annotators=None, pseudo_model_path_func=None, pseudo_func_args={},
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.label_dim = label_dim
        self.annotator_dim = annotator_dim
        self.momentum = momentum
        self.model_weights_path = model_weights_path
        self.save_path_head = save_path_head
        self.save_at = save_at
        self.save_params = save_params
        self.device = device
        self.writer = writer
        self.verbose = verbose
        self.averaging_method = averaging_method
        self.use_softmax = use_softmax

        # List with pseudo annotators and separate function for getting a model path
        self.pseudo_annotators = pseudo_annotators
        self.pseudo_model_path_func = pseudo_model_path_func
        self.pseudo_func_args = pseudo_func_args

        if pseudo_annotators is not None:
            self._create_pseudo_labels()

        if self.device.type == 'cpu':
            from datasets import collate_wrapper_cpu as collate_wrapper
        elif self.device.type == 'cuda':
            from datasets import collate_wrapper
        self.collate_wrapper = collate_wrapper

    def _get_model(self, basic_only=False, pretrained_basic=False):
        if not basic_only:
            model = Ipa2ltHead(self.embedding_dim, self.label_dim, self.annotator_dim, use_softmax=self.use_softmax)
        else:
            model = BasicNetwork(self.embedding_dim, self.label_dim, use_softmax=self.use_softmax)
        if self.model_weights_path is not '':
            if self.verbose:
                print(
                    f'Training model with weights of file {self.model_weights_path}')
            if pretrained_basic and not basic_only:
                model.basic_network.load_state_dict(torch.load(self.model_weights_path))
            else:
                model.load_state_dict(torch.load(self.model_weights_path))
        model.to(self.device)

        return model

    def _create_pseudo_labels(self):
        model = BasicNetwork(self.embedding_dim, self.label_dim, use_softmax=self.use_softmax)
        for pseudo_ann in self.pseudo_annotators:
            model.load_state_dict(torch.load(self.pseudo_model_path_func(**self.pseudo_func_args, annotator=pseudo_ann)))
            model.to(self.device)
            annotator_list = self.dataset.annotators.copy()
            annotator_list.remove(pseudo_ann)
            for annotator in annotator_list:
                self.dataset.create_pseudo_labels(annotator, pseudo_ann, model)

    def _print(self, *args, **kwargs):

        print(*args, **kwargs)

    def fit(self, epochs, return_f1=False, single_annotator=None, basic_only=False, fix_base=False, pretrained_basic=False, deep_randomization=False):
        model = self._get_model(basic_only=basic_only, pretrained_basic=pretrained_basic)
        if single_annotator is not None or basic_only:
            self.annotator_dim = 1
            optimizers = [optim.AdamW([
                {'params': model.parameters()},
            ], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)]

        # if self.label_dim is 2:
        #    criterion = nn.BCELoss()
        # elif self.label_dim > 2:
        criterion = nn.CrossEntropyLoss()
        if single_annotator is None and not basic_only:
            if not fix_base:
                optimizers = [optim.AdamW([
                    {'params': model.basic_network.parameters()},
                    {'params': model.bias_matrices[i].parameters()},
                ],
                    lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                    for i in range(self.annotator_dim)]
            else:
                optimizers = [optim.AdamW([
                    {'params': model.bias_matrices[i].parameters()},
                ],
                    lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
                    for i in range(self.annotator_dim)]

        loss_history = []
        inputs = 0
        labels = 0

        # self._print('START TRAINING')
        if self.verbose:
            self._print(
                f'learning rate: {self.learning_rate} - batch size: {self.batch_size}')
        for epoch in range(epochs):
            f1 = 0.0
            samples_looked_at = 0.0

            if deep_randomization:
                if single_annotator is not None:
                    self.dataset.set_annotator_filter(single_annotator)
                    annotators = [single_annotator]
                else:
                    self.dataset.no_annotator_filter()
                    annotators = self.dataset.annotators

                # training
                self.dataset.set_mode('train')
                train_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper, shuffle=True)
                self.fit_epoch_deep_randomization(model, optimizers, criterion, train_loader, epoch, loss_history,
                                                  annotators=annotators, basic_only=basic_only)
                # validation
                self.dataset.set_mode('validation')
                val_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper, shuffle=True)
                _, _, f1 = self.fit_epoch_deep_randomization(model, optimizers, criterion, val_loader, epoch,
                                                             loss_history, annotators=annotators,
                                                             basic_only=basic_only, mode='validation', return_metrics=return_f1)
                if f1 is not None and isinstance(f1, dict):
                    f1 = sum(f1.values()) / len(f1.values())

            else:
                # loop over all annotators
                for i in range(self.annotator_dim):
                    # switch to current annotator
                    if single_annotator is not None:
                        annotator = single_annotator
                        self.dataset.set_annotator_filter(annotator)
                        no_annotator_head = True
                        optimizer = optimizers[0]
                    elif single_annotator is None and basic_only:
                        annotator = 'all'
                        self.dataset.no_annotator_filter()
                        no_annotator_head = True
                        optimizer = optimizers[0]
                    else:
                        annotator = self.dataset.annotators[i]
                        optimizer = optimizers[i]
                        self.dataset.set_annotator_filter(annotator)
                        no_annotator_head = False

                    # training
                    self.dataset.set_mode('train')
                    train_loader = torch.utils.data.DataLoader(
                        self.dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper)
                    self.fit_epoch(model, optimizer, criterion, train_loader, annotator, i,
                                   epoch, loss_history, no_annotator_head=no_annotator_head)

                    # validation
                    self.dataset.set_mode('validation')
                    val_loader = torch.utils.data.DataLoader(
                        self.dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper)
                    if return_f1:
                        if len(val_loader) is 0:
                            self.dataset.set_mode('train')
                            val_loader = torch.utils.data.DataLoader(
                                self.dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper)
                        _, _, f1_ann = self.fit_epoch(model, optimizer, criterion, val_loader, annotator, i,
                                                      epoch, loss_history, mode='validation', return_metrics=True,
                                                      no_annotator_head=no_annotator_head)
                        # essentially micro averaging across annotators
                        f1 = (samples_looked_at * f1 + f1_ann * len(self.dataset)) / (samples_looked_at + len(self.dataset))
                        samples_looked_at += len(self.dataset)
                        # print(f'DEBUG SOLVER - i: {i}, f1: {f1}, f1_ann: {f1_ann}, samples_looked_at: {samples_looked_at},
                        # len dataset: {len(self.dataset)}')
                    else:
                        self.fit_epoch(model, optimizer, criterion, val_loader, annotator, i,
                                       epoch, loss_history, mode='validation', no_annotator_head=no_annotator_head)

            if self.save_at is not None and self.save_path_head is not None and self.save_params is not None:
                if epoch in self.save_at:
                    params = self.save_params
                    if return_f1:
                        path = get_model_path(self.save_path_head, params['stem'], params['current_time'], params['hyperparams'], f1)
                    else:
                        path = get_model_path(self.save_path_head, params['stem'], params['current_time'], params['hyperparams'])
                    path += f'_epoch{epoch}.pt'

                    print(f'Saving model at: {path}')
                    torch.save(model.state_dict(), path)

        if self.verbose:
            self._print('Finished Training' + 20 * ' ')
            self._print('sum of first 10 losses: ', sum(loss_history[0:10]))
            self._print('sum of last  10 losses: ', sum(loss_history[-10:]))

        if return_f1:
            return model, f1

        return model

    def fit_epoch(self, model, opt, criterion, data_loader, annotator, annotator_idx, epoch, loss_history, mode='train',
                  return_metrics=False, no_annotator_head=False):
        if no_annotator_head:
            annotator_idx = None
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
            inputs, labels, pseudo_labels = data.input, data.target, data.pseudo_targets
            opt.zero_grad()

            # Generate predictions
            if annotator_idx is not None:
                outputs = model(inputs)[annotator_idx]
                if len(pseudo_labels) is not 0:
                    outputs_pseudo_labels = model(inputs)
                    if isinstance(pseudo_labels, list):
                        pseudo_annotators = set([ann for sample in pseudo_labels for ann in list(sample.keys())])
                        losses = [criterion(outputs_pseudo_labels[self.dataset.annotators.index(ann)].float(),
                                            torch.tensor([sample[ann] for sample in pseudo_labels]).to(device=self.device))
                                  for ann in pseudo_annotators]
                    else:
                        losses = [criterion(outputs_pseudo_labels[self.dataset.annotators.index(ann)].float(), pseudo_labels[ann])
                                  for ann in pseudo_labels.keys()]
            else:
                outputs = model(inputs)

            # Compute Loss:
            loss = criterion(outputs.float(), labels)

            # performance measures of the batch
            predictions = outputs.argmax(dim=1)
            accuracy, precision, recall, f1 = self.performance_measures(predictions, labels, self.averaging_method)

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
                if annotator_idx is not None and len(pseudo_labels) is not 0:
                    final_loss = loss
                    for pseudo_loss in losses:
                        final_loss += pseudo_loss
                    final_loss.backward()
                else:
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

    def fit_epoch_deep_randomization(self, model, optimizers, criterion, data_loader, epoch, loss_history, annotators=[], mode='train',
                                     return_metrics=False, basic_only=False):
        """
            this function is made to take a dataset with no annotation filter
            and thus having batches with samples by different annotators. It also
            works with annotation filters. (normal fit_epoch function is more efficient
            though for this purpose)
        """
        # Setup
        if basic_only:
            single_annotator = None
            if len(annotators) is 1:
                single_annotator = annotators[0]
            annotators = []
            mean_loss = 0.0
            mean_accuracy = 0.0
            mean_precision = 0.0
            mean_recall = 0.0
            mean_f1 = 0.0
            if len(optimizers) != 1:
                print('ERROR - please only provide one optimizer when training a basic model only!')
                return
        else:
            if len(annotators) == 0:
                print('ERROR - Please provide annotators in correct order!')
                return

            if len(annotators) != len(optimizers):
                print('ERROR - Please provide as many optimizers as there are annotators!')
                return
            mean_loss = {ann: 0.0 for ann in annotators}
            mean_accuracy = {ann: 0.0 for ann in annotators}
            mean_precision = {ann: 0.0 for ann in annotators}
            mean_recall = {ann: 0.0 for ann in annotators}
            mean_f1 = {ann: 0.0 for ann in annotators}

        # Training loop
        len_data_loader = len(data_loader)
        for i, data in enumerate(data_loader, 1):
            inputs, labels, pseudo_labels, annotations = data.input, data.target, data.pseudo_targets, data.annotations
            for opt in optimizers:
                opt.zero_grad()

            # Generate predictions
            losses = {}
            if len(pseudo_labels) is not 0:
                pseudo_annotators = set([ann for sample in pseudo_labels for ann in list(sample.keys())])

            for annotator_idx, annotator in enumerate(annotators):
                self._print(
                    f'Annotator {annotator} - Epoch {epoch}: Step {i} / {len_data_loader}' + 10 * ' ', end='\r')
                outputs = model(inputs)
                outputs_annotator = outputs[annotator_idx]
                loss_annotations = None
                if annotator in set(annotations):
                    mask_labels = torch.tensor([ann == annotator for ann in annotations]).to(device=self.device)
                    mask_outputs = torch.tensor([[ann == annotator, ann == annotator] for ann in annotations]).to(device=self.device)
                    labels_annotations = torch.masked_select(labels, mask_labels)
                    outputs_dim = (mask_labels[mask_labels].shape[0], outputs_annotator.shape[1])
                    outputs_annotations = torch.masked_select(outputs_annotator, mask_outputs).reshape(outputs_dim)
                    loss_annotations = criterion(outputs_annotations.float(), labels_annotations)

                # search for this annotator in pseudo labels of samples by all other annotators to add the losses
                loss_pseudo_annotations = None
                if len(pseudo_labels) is not 0:
                    if annotator in pseudo_annotators:
                        mask_labels = [annotator in list(sample.keys()) for sample in pseudo_labels]
                        labels_pseudo_annotations = torch.tensor([int(sample[annotator])
                                                                  for sample in compress(pseudo_labels, mask_labels)]).to(device=self.device)
                        mask_outputs = torch.tensor([[annotator in list(sample.keys()), annotator in list(sample.keys())]
                                                     for sample in pseudo_labels]).to(device=self.device)
                        outputs_dim = (len([x for x in compress(mask_labels, mask_labels)]), outputs_annotator.shape[1])
                        outputs_pseudo_annotations = torch.masked_select(outputs_annotator, mask_outputs).reshape(outputs_dim)
                        loss_pseudo_annotations = criterion(outputs_pseudo_annotations.float(), labels_pseudo_annotations)

                if loss_annotations is not None or loss_pseudo_annotations is not None:
                    if loss_annotations is not None and loss_pseudo_annotations is not None:
                        loss = loss_annotations + loss_pseudo_annotations
                    elif loss_annotations is not None and loss_pseudo_annotations is None:
                        loss = loss_annotations
                    elif loss_pseudo_annotations is not None and loss_annotations is None:
                        loss = loss_pseudo_annotations

                    if annotator in set(annotations):
                        # record performance for this annotator (discard pseudo annotations)
                        predictions = outputs_annotations.argmax(dim=1)
                        accuracy, precision, recall, f1 = self.performance_measures(predictions, labels_annotations, self.averaging_method)

                        # statistics for logging
                        current_batch_size = inputs.shape[0]
                        divisor = (i - 1) * self.batch_size + current_batch_size
                        mean_loss[annotator] = ((i - 1) * self.batch_size * mean_loss[annotator] +
                                                loss.item() * current_batch_size) / divisor
                        mean_accuracy[annotator] = (mean_accuracy[annotator] * self.batch_size * (i - 1) +
                                                    accuracy.item() * current_batch_size) / divisor
                        mean_precision[annotator] = (mean_precision[annotator] * self.batch_size * (i - 1) +
                                                     precision.item() * current_batch_size) / divisor
                        mean_recall[annotator] = (mean_recall[annotator] * self.batch_size * (i - 1) + recall.item() * current_batch_size) / divisor
                        mean_f1[annotator] = (mean_f1[annotator] * self.batch_size * (i - 1) + f1.item() * current_batch_size) / divisor
                        loss_history.append(loss.item())

                        if self.writer is not None:
                            self.writer.add_scalar(f'Loss/Annotator {annotator}/{mode}', mean_loss[annotator], epoch)
                            self.writer.add_scalar(
                                f'Accuracy/Annotator {annotator}/{mode}', mean_accuracy[annotator], epoch)
                            self.writer.add_scalar(
                                f'Precision/Annotator {annotator}/{mode}', mean_precision[annotator], epoch)
                            self.writer.add_scalar(f'Recall/Annotator {annotator}/{mode}', mean_recall[annotator], epoch)
                            self.writer.add_scalar(f'F1 score/Annotator {annotator}/{mode}', mean_f1[annotator], epoch)

                    if mode is 'train':
                        # Update gradients
                        if loss_annotations is not None:
                            retain_graph = False
                            if loss_pseudo_annotations is not None:
                                retain_graph = True
                            loss_annotations.backward(retain_graph=retain_graph)
                        if loss_pseudo_annotations is not None:
                            loss_pseudo_annotations.backward()

                        # Optimization step
                        optimizers[annotator_idx].step()
                        optimizers[annotator_idx].zero_grad()

            if basic_only:
                annotator = 'all'
                if single_annotator is not None:
                    annotator = single_annotator
                self._print(
                    f'Annotator {single_annotator} - Epoch {epoch}: Step {i} / {len_data_loader}' + 10 * ' ', end='\r')
                outputs = model(inputs)

                # Compute Loss:
                loss = criterion(outputs.float(), labels)

                # performance measures of the batch
                predictions = outputs.argmax(dim=1)
                accuracy, precision, recall, f1 = self.performance_measures(predictions, labels, self.averaging_method)

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
                    optimizers[0].step()

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

    def evaluate_model(self, output_file_path, labels=None, mode='train', pretrained_basic_path=''):
        model = self._get_model()

        # load pretrained model for comparison
        if pretrained_basic_path != '':
            pretrained_model = BasicNetwork(self.embedding_dim, self.label_dim, use_softmax=self.use_softmax)
            pretrained_model.load_state_dict(torch.load(pretrained_basic_path))
            pretrained_model.to(self.device)

        # write annotation bias matrices into log file
        import sys
        original_stdout = sys.stdout
        with open(output_file_path, 'w') as f:
            sys.stdout = f
            self.dataset.set_mode(mode)
            out_text = ''
            acc_out = ''
            bias_out = 'Annotation bias matrices\n\n'
            overall_correct = 0
            overall_len = 0
            if pretrained_basic_path != '':
                pretrained_correct = 0
            for i, annotator in enumerate(self.dataset.annotators):
                bias_out += f'Annotator {annotator}\n'
                bias_out += f'Output\\LatentTruth'
                if labels is not None:
                    for label in labels:
                        bias_out += '\t' * 3 + f'{label}'
                    bias_out += '\n'
                    for j, label in enumerate(labels):
                        bias_out += f'{label}' + ' ' * (15 - len(label))
                        for k, label_2 in enumerate(labels):
                            bias_out += '\t' * 3 + f'{model.bias_matrices[i].weight[j][k].cpu().detach().numpy(): .4f}'
                        bias_out += '\n'
                    bias_out += '\n'
                else:
                    bias_out += f'{model.bias_matrices[i].weight.cpu().detach().numpy()}\n\n'

                correct = {ann: 0 for ann in self.dataset.annotators}

                different_answers = 0
                different_answers_idx = []

                self.dataset.set_annotator_filter(annotator)
                # batch_size needs to be 1
                val_loader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=1, collate_fn=self.collate_wrapper)

                for i, data in enumerate(val_loader, 1):
                    # Prepare inputs to be passed to the model
                    inp, label = data.input, data.target

                    # Generate predictions
                    latent_truth = model.basic_network(inp)
                    output = model(inp)
                    if pretrained_basic_path != '':
                        pretrained_output = pretrained_model(inp)

                    # print input/output
                    out_text += f'Point {i} - Label by {annotator}: {label.cpu().numpy()} - Latent truth {latent_truth.cpu().detach().numpy()}'
                    predictions = {}
                    for idx, ann in enumerate(self.dataset.annotators):
                        out_text += f' - Annotator {ann} {output[idx].cpu().detach().numpy()}'
                        predictions[ann] = [output[idx].argmax(dim=1), output[idx].max(dim=1)]

                    # compare the prediction with the label for each annotator
                    for idx, ann in enumerate(self.dataset.annotators):
                        if predictions[ann][0] == label:
                            # annotator_highest_pred = max(filtered_preds, key=filtered_preds.get)
                            correct[ann] += 1

                    # compare the prediction with the label for the annotator that created the label
                    if predictions[annotator][0] == label:
                        overall_correct += 1

                    # compare the prediction with the label for the pretrained model
                    if pretrained_basic_path != '':
                        if pretrained_output.argmax(dim=1) == label:
                            pretrained_correct += 1

                    predictions_set = set([predictions[ann][0].cpu().detach().numpy().item(0) for ann in self.dataset.annotators])
                    if len(predictions_set) is not 1:
                        different_answers += 1
                        different_answers_idx.append(i)

                    out_text += '\n'

                overall_len += len(self.dataset)

                # Document correct predictions
                all_same_accuracies = set([correct[ann] for ann in self.dataset.annotators])
                acc_out += '-' * 25 + f'   Annotator {annotator}   ' + '-' * 25 + '\n'
                acc_out += f'Different answers given by bias matrices {different_answers} / {len(self.dataset)} times\n'
                acc_out += f'Different answers at points: {different_answers_idx[:min(5, len(different_answers_idx))]}\n'
                acc_out += f'Accuracies of samples labeled by {annotator}:'
                # ' has {len(all_same_accuracies)} '
                # if len(all_same_accuracies) is 1:
                #     acc_out += 'answer\n'
                # else:
                #     acc_out += 'different answers\n'
                acc_out += '\n'
                for ann in self.dataset.annotators:
                    acc_out += f'Annotator {ann}: {correct[ann]} / {len(self.dataset)}     '
                acc_out += '\n\n'

            # Document overall accuracy
            overall_acc_out = 'Overall accuracies\n\n'
            overall_accuracy = overall_correct / overall_len
            overall_acc_out += f'Accuracy with bias matrices: {overall_correct} / {overall_len} or as percentage: {overall_accuracy:.5f}\n'
            if pretrained_basic_path != '':
                pretrained_accuracy = pretrained_correct / overall_len
                overall_acc_out += f'Accuracy with pretrained model: {pretrained_correct} / {overall_len} ' + \
                    f'or as percentage: {pretrained_accuracy:.5f}\n\n'
            else:
                overall_acc_out += '\n\n'

            print(overall_acc_out)
            print(bias_out)
            print(acc_out)
            print('\n' * 30)
            print(out_text)
            sys.stdout = original_stdout

    @staticmethod
    def performance_measures(predictions, labels, averaging_method='macro'):
        if predictions.device.type == 'cuda' or labels.device.type == 'cuda':
            predictions, labels = predictions.cpu(), labels.cpu()

        # averaging for multiclass targets, can be one of [‘micro’, ‘macro’, ‘samples’, ‘weighted’]
        accuracy = accuracy_score(labels, predictions)
        zero_division = 0
        precision = precision_score(labels, predictions, average=averaging_method, zero_division=zero_division)
        recall = recall_score(labels, predictions, average=averaging_method, zero_division=zero_division)
        f1 = f1_score(labels, predictions, average=averaging_method, zero_division=zero_division)

        return accuracy, precision, recall, f1
