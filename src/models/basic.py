from .utils import initialize_weight

import torch.nn as nn


class BasicNetwork(nn.Module):
    def __init__(self, embedding_dim, label_dim, use_softmax=True):
        super().__init__()

        self.attention = nn.Linear(embedding_dim, 1, bias=False)
        self.classifier = nn.Linear(embedding_dim, label_dim)
        self.softmax_batch = nn.Softmax(dim=1)
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()

        self.use_softmax = use_softmax

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

        if self.use_softmax:
            if shape_len is 3:
                x = self.softmax_batch(x)
            elif shape_len is 2:
                x = self.softmax(x)
        else:
            x = self.sigmoid(x)

        return x
