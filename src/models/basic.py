from .utils import initialize_weight

import torch.nn as nn


class BasicNetwork(nn.Module):
    def __init__(self, word_dim, label_dim):
        super().__init__()

        self.attention = nn.Linear(word_dim, 1, bias=False)
        self.classifier = nn.Linear(word_dim, label_dim)
        self.sigmoid = nn.Sigmoid()

        self.apply(initialize_weight)

    def forward(self, x):

        shape_len = len(x.shape)

        # sum up word vectors weighted by their word-wise attentions
        attentions = self.attention(x)
        x = attentions * x

        # sum over all words in x
        if shape_len is 3:
            x = x.sum(axis=1)
        elif shape_len is 2:
            x = x.sum(axis=0)

        # feed it to the classifier
        x = self.classifier(x)

        # apply sigmoid
        x = self.sigmoid(x)

        return x
