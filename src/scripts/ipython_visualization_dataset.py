from torch.utils.tensorboard import SummaryWriter
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import torch
import pytz
import datetime
import matplotlib.pyplot as plt
import numpy as np

from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from utils import *

DEVICE = torch.device('cuda')
LOCAL_FOLDER = '../visuals/datasets'
PNG_NAME = 'sentence_lengths'

# # #  Setup  # # #
# # Parameters dependent on dataset # #

# label_dim = 2
# annotator_dim = 2
# loss = 'nll'
# dataset_name = 'wikipedia'
# task = 'toxicity'
# percentage = 0.05
# group_by_gender = True
# only_male_female = True
# dataset = WikipediaDataset(device=DEVICE, task=task, group_by_gender=group_by_gender,
#                            percentage=percentage, only_male_female=only_male_female)

label_dim = 3
annotator_dim = 10
loss = 'nll'
padding_length = 136
predict_coarse_attributes_task = False
dataset_name = 'organic'
domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
                         padding_length=padding_length, domain_embedding_path=domain_embedding_path)
task = 'class_dist'
extra = ''
if predict_coarse_attributes_task:
    extra = 'coarse_attributes'
if domain_embedding_path is not '':
    extra += '_fine_tuned_emb'
colors = ['indigo', 'lemonchiffon', 'darkseagreen']
labels = {'0': 'negative', '1': 'neutral', '2': 'positive'}

# label_dim = 3
# annotator_dim = 38
# loss = 'nll'
# dataset_name = 'emotion'
# domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
# dataset = EmotionDataset(device=DEVICE, domain_embedding_path=domain_embedding_path)
# emotion = 'valence'
# dataset.set_emotion(emotion)
# task = 'class_dist'
# extra = 'step_plot'
# if domain_embedding_path is not '':
#     extra += '_fine_tuned_emb'
# colors = ['slategrey', 'lightcoral', 'lightsteelblue']
# labels = {'0': 'negative', '1': 'neutral', '2': 'positive'}

# label_dim = 2
# annotator_dim = 2
# loss = 'nll'
# one_dataset_one_annotator = False
# dataset = TripAdvisorDataset(device=DEVICE, one_dataset_one_annotator=one_dataset_one_annotator)
# dataset_name = 'tripadvisor'
# task = 'class_dist'
# extra = 'final'
# colors = ['darkgreen', 'navajowhite']
# labels = {'0': 'negative', '1': 'positive'}
# if one_dataset_one_annotator:
#     task = 'annotator_dist'
#     extra = 'one_dataset_one_annotator'
#     colors = ['darkorange', 'slategrey']
#     labels = {'hotels': 'hotels', 'restaurants': 'restaurants'}


# png path
png_path = f'{LOCAL_FOLDER}/{dataset_name}/{PNG_NAME}_{task}'
if extra is not '':
    png_path += f'_{extra}'
png_path += '.pdf'

# iterate over
distribution = [i for i in range(label_dim)]
if task is 'annotator_dist':
    distribution = dataset.annotators

# calculate sentence length for every sample
tokenizer = RegexpTokenizer(r'\w+')
text_lengths = {f'{elem}': [] for elem in distribution}
for elem in distribution:
    samples_text = []
    for mode in ['train', 'validation', 'test']:
        dataset.set_mode(mode)
        if task is 'class_dist':
            samples_text.extend([sample['text'] for sample in dataset if sample['label'].item() == elem])
        if task is 'annotator_dist':
            samples_text.extend([sample['text'] for sample in dataset if sample['annotator'] == elem])
    samples_text = set(samples_text)
    text_lengths[f'{elem}'] = [len(tokenizer.tokenize(sample)) for sample in samples_text]

# visualize metrics side by side, oriented at https://python-graph-gallery.com/11-grouped-barplot/
if dataset_name == 'tripadvisor':
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
elif dataset_name == 'emotion':
    fig, axes = plt.subplots(ncols=3, figsize=(21, 6))
elif dataset_name == 'organic':
    fig, axes = plt.subplots(nrows=3, figsize=(6, 6))


# bar_width = 3
# negative_bars = dict(sorted(dict(Counter(text_lengths['0'])).items()))
# positive_bars = dict(sorted(dict(Counter(text_lengths['1'])).items()))

# pos1 = negative_bars.keys()   # np.arange(len(glove_bars))
# pos2 = [pos + bar_width for pos in positive_bars.keys()]

# plt.bar(pos1, negative_bars.values(), color='coral', width=bar_width, edgecolor='white', label='negative')
# plt.bar(pos2, positive_bars.values(), color='lightsteelblue', width=bar_width, edgecolor='white', label='positive')

xlim = max([max(text_lengths[f'{elem}']) for elem in distribution])
ylim = max([max(np.unique(text_lengths[f'{elem}'], return_counts=True)[1]) for elem in distribution]) + 10
for i, elem in enumerate(distribution):
    nbins = max(text_lengths[f'{elem}']) - min(text_lengths[f'{elem}'])
    ax = axes.flatten()[i]
    sample_len = len(text_lengths[f'{elem}'])
    label = labels[f'{elem}']
    if dataset_name is 'emotion':
        ax.hist(text_lengths[f'{elem}'], nbins, histtype='step', lw=3, color=colors[i], label=f'{label} samples ({sample_len})')
    else:
        ax.hist(text_lengths[f'{elem}'], nbins, histtype='bar', color=colors[i], label=f'{label} samples ({sample_len})')

    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    ax.set_xlabel('Sample length [words]', fontweight='bold')
    ax.set_ylabel('Occurence [samples]', fontweight='bold')

    ax.legend(fontsize=10)

fig.tight_layout()
plt.savefig(png_path)
