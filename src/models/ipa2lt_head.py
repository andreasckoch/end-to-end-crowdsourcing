from models.utils import initialize_weight
from models.basic import BasicNetwork
from models.transformers import model

import torch.nn as nn
import torch


class Ipa2ltHead(nn.Module):
    def __init__(self, embedding_dim, label_dim, annotator_dim):
        super().__init__()

        self.annotator_dim = annotator_dim
        self.label_dim = label_dim
        self.basic_network = BasicNetwork(embedding_dim, label_dim)
        self.bias_matrices = nn.ModuleList([nn.Linear(label_dim, label_dim, bias=False) for i in range(annotator_dim)])

        self.apply(initialize_weight)

    def forward(self, x):

        #x = self.basic_network(x)
        x = model.forward(x)
        out = []
        for matrix in self.bias_matrices:
            # normalize rows of bias matrices
            normalized = matrix.weight / torch.norm(matrix.weight, dim=1, p=1, keepdim=True)
            matrix.weight = nn.Parameter(normalized.abs())

            # forward pass
            pred = matrix(x)
            out.append(pred)

        return out
