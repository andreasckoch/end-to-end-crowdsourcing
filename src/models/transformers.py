import torch
from transformers import LongformerModel, LongformerTokenizer

model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True)
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
