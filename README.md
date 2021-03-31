# End-To-End Crowdsourcing
Comparison of traditional crowdsourcing approaches to a state-of-the-art end-to-end crowdsourcing approach LTNet on sentiment analysis.
LTNet is adapted from ["Facial Expression Recognition with Inconsistently Annotated Datasets"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jiabei_Zeng_Facial_Expression_Recognition_ECCV_2018_paper.pdf) to text data. It encompasses a simple attention based neural network and utilizes confusion matrices as a noise reduction technique. For comparison, the traditional ground truth estimators ["Fast-Dawid-Skene"](https://arxiv.org/pdf/1803.02781) and ["MACE"](www.cs.cmu.edu/~./hovy/papers/13HLT-MACE.pdf) are applied. 

This codebase was used in both ["End-to-End Annotator Bias Approximation on Crowdsourced Single-Label Sentiment Analysis"]() and ["Deep End-to-End Learning for Noisy Annotations and Crowdsourcing in Natural Language Processing"]().

## Training

This is an example training procedure for the TripAdvisor dataset.
The dataset and solver objects are initialized before a standard LTNet model is trained for 300 epochs.
```python
import torch
import pytz
import datetime

from datasets.tripadvisor import TripAdvisorDataset
from solver import Solver
from utils import *

# gpu
DEVICE = torch.device('cuda')

# cpu
# DEVICE = torch.device('cpu')

label_dim = 2
annotator_dim = 2
loss = 'nll'
one_dataset_one_annotator = False
dataset = TripAdvisorDataset(device=DEVICE, one_dataset_one_annotator=one_dataset_one_annotator)

lr = 1e-5
batch_size = 64
current_time = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d-%H%M%S")
hyperparams = {'batch': batch_size, 'lr': lr}
writer = get_writer(path=f'../logs/test',
                    current_time=current_time, params=hyperparams)

solver = Solver(dataset, lr, batch_size, 
                writer=writer,
                device=DEVICE,
                label_dim=label_dim,
                annotator_dim=annotator_dim)

model, f1 = solver.fit(epochs=300, return_f1=True,
                       deep_randomization=True)
```

These initialization and training steps of a network are abstracted away into src/training. Scripts with many more details on training procedures and different configurations can be found in src/scripts. All are best loaded into an ipython terminal with the %load command.


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

Datasets are available at ["TripAdvisor"](https://ndownloader.figshare.com/files/11432270), ["Emotion"](https://sites.google.com/site/nlpannotations/) and ["Organic"]().

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
