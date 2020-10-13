from transformers import LongformerForSequenceClassification, LongformerTokenizer
import torch

print('\n\nWARNING:\n\n')

model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', return_dict = True)
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

sequence = "Testing on the longformer model"
tokenized_sequence = tokenizer.tokenize(sequence)
encoded_sequence = tokenizer.encode(sequence)

inputs = tokenizer(sequence, return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)		# Batch of size 1

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits

print('\n\nPRINTING PROPERTIES:\n\n')

print('sequence:	', sequence)
print('tokenized_seq:	', tokenized_sequence)
print('encoded_seq:	', encoded_sequence)

print('\nlabels:		', str(labels))
print('inputs:		', inputs)
print('*inputs:	', *inputs)
print('loss:		', loss)
print('logits:		', logits)
