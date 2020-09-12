# awesome-tree

## Database

code
```python 
from datasets.tripadvisor import TripAdvisorDataset

dataset = TripAdvisorDataset(text_processor='word2vec', text_processor_filters=['lowercase', 'stopwordsfilter'])

print(f'Dataset is in {dataset.mode} mode')
print(f'Train-Validation split is {dataset.train_val_split}')
print(f'1st train datapoint: {dataset[0]}')
```

output
```
Dataset is in train mode
Train-Validation split is 0.8
1st train datapoint: {'label': 'f', 'rating': 4, 'text': 'I realise ...', 'embedding': array}
```
