import torch
import torch.nn as nn


def initialize_weight(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)

def initialize_bias_matrices(module):
    if isinstance(module, nn.Linear):
        nn.init.eye_(module.weight)
        module.weight = nn.Parameter(module.weight + torch.rand(module.weight.shape) * 0.1)
        normalized = module.weight / torch.norm(module.weight, dim=1, p=1, keepdim=True)
        module.weight = nn.Parameter(normalized.abs())
