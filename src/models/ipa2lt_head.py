from .utils import initialize_weight, initialize_bias_matrices
from .basic import BasicNetwork

import torch.nn as nn
import torch


class Ipa2ltHead(nn.Module):
    def __init__(self, embedding_dim, label_dim, annotator_dim, use_softmax=True, apply_log=False):
        super().__init__()

        self.annotator_dim = annotator_dim
        self.label_dim = label_dim
        self.apply_log = apply_log
        self.basic_network = BasicNetwork(embedding_dim, label_dim, use_softmax=use_softmax)
        self.bias_matrices = nn.ModuleList([nn.Linear(label_dim, label_dim, bias=False) for i in range(annotator_dim)])

        self.basic_network.apply(initialize_weight)
        self.bias_matrices.apply(initialize_bias_matrices)

    def forward(self, x):

        x = self.basic_network(x)
        out = []
        for matrix in self.bias_matrices:
            # normalize rows of bias matrices
            normalized = matrix.weight / torch.norm(matrix.weight, dim=1, p=1, keepdim=True)
            matrix.weight = nn.Parameter(normalized.abs())

            # forward pass
            pred = torch.matmul(x.T, matrix.weight)
            if self.apply_log:
                pred = torch.clamp(torch.log(torch.clamp(pred, 1e-5)), -100.0)
            out.append(pred)

        return out
