# https://huggingface.co/transformers/v2.10.0/model_doc/longformer.html

import torch
from transformers import LongformerModel, LongformerTokenizer

model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True)
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

SAMPLE_TEXT = ' '.join(['God is dead. '] * 1000)  # long input document
input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

# Attention mask values -- 0: no attention, 1: local attention, 2: global attention
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # 
global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
global_attention_mask[:, [1, 4, 21,]] = 1
outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
sequence_output = outputs.last_hidden_state
pooled_output = outputs.pooler_output


