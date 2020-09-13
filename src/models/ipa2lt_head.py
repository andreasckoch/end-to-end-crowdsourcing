from .utils import initialize_weight
from .basic import BasicNetwork

import torch.nn as nn


class Ipa2ltHead(nn.Module):
    def __init__(self, word_dim, label_dim):
        super().__init__()

        self.label_dim = label_dim
        self.basic_network = BasicNetwork(word_dim, label_dim)
        self.bias_matrices = nn.ModuleList([nn.Linear(label_dim, label_dim, bias=False) for i in range(label_dim)])
        self.sigmoid = nn.Sigmoid()

        self.apply(initialize_weight)

    def forward(self, x):

        x = self.basic_network(x)

        out = []
        for matrix in self.bias_matrices:
            pred = matrix(x)
            pred = self.sigmoid(pred)
            out.append(pred)

        return out
