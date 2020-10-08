
import torch
import numpy as np
from datasets.tripadvisor import TripAdvisorDataset
from transformers import LongformerModel, LongformerTokenizer

dataset = TripAdvisorDataset(text_processor='word2vec', text_processor_filters=['lowercase', 'stopwordsfilter'])
model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True)
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

print(f'Dataset is in {dataset.mode} mode')
print(f'Train-Validation split is {dataset.train_val_split}')
print('1st train datapoint:', dataset[0])

text = []
for i in range(len(dataset)):
    text.append(dataset[i]['text'])
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
print(input_ids)





"""
def fit(self, epochs, return_f1=False):
        
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True)
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


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

        self._print('START TRAINING')
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

    return model"""