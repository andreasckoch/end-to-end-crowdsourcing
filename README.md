# awesome-tree

## Databases

#### How to use them from outside the src folder?

It makes us able to refer to the classes properly.
```python
import sys
sys.path.append("src/")
```

Pass the root folders of the embeddings and the data. 

```python
from datasets.emotion import EmotionDataset

dataset = EmotionDataset(
        text_processor='word2vec', 
        text_processor_filters=['lowercase', 'stopwordsfilter'],
        embedding_path='data/embeddings/word2vec/glove.6B.50d.txt',
        data_path='data/'
        )
```

### TripAdvisor Dataset

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
1st train datapoint: {'label': 0, 'annotator':'f', 'rating': 4, 'text': 'I realise ...', 'embedding': array}
```



### Emotion Dataset

Every headline has been annotated on each emotion. 
One can select one emotion as the `label` by the `set_emotion` method.

code
```python 
from datasets.emotion import EmotionDataset

dataset = TripAdvisorDataset(text_processor='word2vec', text_processor_filters=['lowercase', 'stopwordsfilter'])

print(f'Dataset is in {dataset.mode} mode')
print(f'Train-Validation split is {dataset.train_val_split}')
dataset.set_emotion('anger')
print(f'1st train datapoint: {dataset[0]}') # select anger_label as label
dataset.set_emotion('disgust')
print(f'1st train datapoint: {dataset[0]}') # select disgust_label as label
```

output
```
Dataset is in train mode
Train-Validation split is 0.8
1st train datapoint: {'label': 0, 'annotator':'xxx1', 'anger_response':0, 'anger_label':0, 'anger_gold'=1, 'disgust_response':0 ... 'text': 'I realise ...', ... 'embedding': array}
1st train datapoint: {'label': 1, 'annotator':'xxx1', 'anger_response':0, 'anger_label':0, 'anger_gold'=1, 'disgust_response':0 ... 'text': 'I realise ...', ... 'embedding': array}
```
