from .utils import initialize_weight
from transformers import LongformerModel
import torch
import torch.nn as nn


class BasicNetwork(nn.Module):
    def __init__(self, embedding_dim, label_dim):
        super().__init__()

        self.attention = nn.Linear(embedding_dim, 1, bias=False)
        self.classifier = nn.Linear(embedding_dim, label_dim)
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

        """
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        output = model(input_ids=x)
        x = output.last_hidden_state	# sequence_output
        """

        return x
