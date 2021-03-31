from torch.utils.tensorboard import SummaryWriter
from nltk.metrics.agreement import AnnotationTask
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import torch
import pytz
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.tripadvisor import TripAdvisorDataset
from datasets.emotion import EmotionDataset
from datasets.wikipedia import WikipediaDataset
from datasets.organic import OrganicDataset
from models.ipa2lt_head import Ipa2ltHead
from solver import Solver
from utils import *

DEVICE = torch.device('cuda')
LOCAL_FOLDER = 'train_02_14/sgd/nll'
VISUALS_FOLDER = '../visuals/crowdsourcing'

EPOCHS = 300
USE_EPOCH_FACTOR = True
NUM_DRAWS = 20

# [10, 100, 300], [10, 100, 200], [10, 100, 200]]
SAVE_MODEL_AT = [10, 100, 200, 500, 1000, 2000]
EARLY_STOPPING_INTERVAL = 10
EARLY_STOPPING_MARGIN = 1e-5
# MODEL_WEIGHTS_PATH = '../models/train_10_17/tripadvisor/pretraining_softmax/' + \
#     '0.88136_batch64_lr0.00031867445707134466_20201019-092128_epoch300.pt'

USE_SOFTMAX = True
# macro doesn't make too much sense since we're calculating f1 after every batch --> micro
AVERAGING_METHOD = 'micro'
LR_INT = [1e-6, 1e-3]
BATCH_SIZES = [64]
DEEP_RANDOMIZATION = True
OPTIMIZER = 'sgd'

# # #  Setup  # # #
# # Parameters dependent on dataset # #

# label_dim = 2
# annotator_dim = 2
# loss = 'nll'
# dataset_name = 'wikipedia'
# dataset_name_map = dataset_name
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
dataset_name_map = dataset_name
no_shuffle = False
domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
dataset = OrganicDataset(device=DEVICE, predict_coarse_attributes_task=predict_coarse_attributes_task,
                         padding_length=padding_length, domain_embedding_path=domain_embedding_path, no_shuffle=no_shuffle)
task = 'sentiment'
extra = ''
if predict_coarse_attributes_task:
    extra = 'coarse_attributes'
if domain_embedding_path is not '':
    extra += '_fine_tuned_emb'
    task += '_fine_tuned_emb'
colors = ['indigo', 'lemonchiffon', 'indigo']
labels = {'0': 'negative', '1': 'neutral', '2': 'positive'}
model_path = 'ltnet_pseudo/0.44868_batch64_lr7.084359020902504e-05_20210215-042823_epoch200.pt'

# label_dim = 3
# annotator_dim = 38
# loss = 'nll'
# dataset_name = 'emotion'
# dataset_name_map = dataset_name
# no_shuffle = True
# domain_embedding_path = f'../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl'
# dataset = EmotionDataset(
#     device=DEVICE, domain_embedding_path=domain_embedding_path, no_shuffle=no_shuffle)
# emotion = 'valence'
# dataset.set_emotion(emotion)
# task = 'valence'
# extra = 'step_plot'
# if domain_embedding_path is not '':
#     extra += '_fine_tuned_emb'
#     task += '_fine_tuned_emb'
# colors = ['slategrey', 'lightcoral', 'indigo']
# labels = {'0': 'negative', '1': 'neutral', '2': 'positive'}
# model_path = 'ltnet_true/0.43333_batch64_lr5.845444554359771e-06_20210314-014047_epoch100.pt'

# label_dim = 2
# annotator_dim = 2
# loss = 'nll'
# one_dataset_one_annotator = False
# no_shuffle = False
# dataset = TripAdvisorDataset(device=DEVICE, one_dataset_one_annotator=one_dataset_one_annotator, no_shuffle=no_shuffle)
# dataset_name = 'tripadvisor'
# dataset_name_map = dataset_name
# task = 'gender'
# extra = 'white_edge'
# colors = ['darkslategray', 'darkorange']
# labels = {'0': 'negative', '1': 'positive'}
# if one_dataset_one_annotator:
#     dataset_name_map += '/1.3'
#     task = 'dataset'
#     extra = 'one_dataset_one_annotator'
#     colors = ['darkgreen', 'navajowhite']
#     labels = {'hotels': 'hotels', 'restaurants': 'restaurants'}
#     model_path = 'ltnet_true/0.89077_batch64_lr2.673184899136622e-06_20210217-120352_epoch10.pt'
# else:
#     dataset_name_map += '/1.2'
#     model_path = 'ltnet_pseudo/0.88156_batch64_lr9.179617669307897e-06_20210216-023727_epoch10.pt'


model_path = f'../models/{LOCAL_FOLDER}/{dataset_name}/{task}/{model_path}'


def sample_label_map(dataset_name, method):
    return f"../data/{method}/{dataset_name}/sample_label_map_train.pkl"


# load all dawid_skene/mace labels
dawid_skene_labels = {}
with open(sample_label_map(dataset_name_map, 'dawid_skene'), 'rb') as f:
    dawid_skene_labels = pickle.load(f)
mace_labels = {}
with open(sample_label_map(dataset_name_map, 'mace'), 'rb') as f:
    mace_labels = pickle.load(f)

# load psuedo labels for majority voting
solver_params = {
    'device': DEVICE,
    'label_dim': label_dim,
    'annotator_dim': annotator_dim,
    'averaging_method': AVERAGING_METHOD,
    'use_softmax': USE_SOFTMAX,
    'loss': loss,
    'optimizer_name': OPTIMIZER,
    'early_stopping_margin': EARLY_STOPPING_MARGIN,
    'save_at': SAVE_MODEL_AT,
}
if dataset_name is 'tripadvisor' or dataset_name is 'organic':
    pseudo_root = f'../models/{LOCAL_FOLDER}/{dataset_name}/{task}'
    pseudo_func_args = {
        'pseudo_root': pseudo_root,
        'phase': 'individual_training',
    }
    pseudo_model_path_func = get_pseudo_model_path

    solver_params.update({
        'pseudo_annotators': dataset.annotators,
        'pseudo_model_path_func': pseudo_model_path_func,
        'pseudo_func_args': pseudo_func_args,
    })
    # solver needed only for pseudo labels
    solver = Solver(dataset, 1e-5, BATCH_SIZES[0],
                    **solver_params)

# majority voting labels
mv_labels = {sample: [] for sample in set(dawid_skene_labels.keys())}
for mode in list(dataset.data.keys()):
    dataset.set_mode(mode)
    for sample in dataset:
        if sample['text'] in list(mv_labels.keys()):
            mv_labels[sample['text']].append(sample['label'].item())
            # sample_text = 'Bananaconda inventor is top poet of kids'
            # if sample['text'] == sample_text:
            #     annotator = sample['annotator']
            #     label = sample['label'].item()
            #     print(f'{sample_text} - {annotator}: {label}')
            for pseudo_label in list(sample['pseudo_labels'].values()):
                mv_labels[sample['text']].append(pseudo_label.item())

mv_labels = {sample: Counter(mv_labels[sample]).most_common(1)[
    0][0] for sample in list(mv_labels.keys())}

# load LTNet classifier
model = Ipa2ltHead(50, label_dim, annotator_dim, use_softmax=USE_SOFTMAX)
model.to(DEVICE)
model.load_state_dict(torch.load(model_path))
basic = model.basic_network

ltnet_labels = {}
for mode in list(dataset.data.keys()):
    dataset.set_mode(mode)
    for sample in dataset:
        if sample['text'] in list(dawid_skene_labels.keys()):
            ltnet_labels[sample['text']] = basic(
                sample['embedding']).argmax().item()

# filter redundant samples and possible samples not in dawid_skene map
ltnet_labels = {sample: ltnet_labels[sample]
                for sample in set(dawid_skene_labels.keys())}

# get correct values
big_samples_labels_map = {sample: {
    'ds': dawid_skene_labels[sample],
    'mace': mace_labels[sample],
    'mv': mv_labels[sample],
    'ltnet': ltnet_labels[sample],
} for sample in set(dawid_skene_labels.keys()) if sample in set(mace_labels.keys())}


# krippendorf alpha
def krippendorf_alpha(annotations):
    t = AnnotationTask(annotations)  # distance=binary_distance per default
    return t.alpha()  # Krippendorff's alpha


overlap = {
    'All match': [item[0] for item in list(big_samples_labels_map.items())
                  if item[1]['ds'] == item[1]['mace'] == item[1]['mv'] == item[1]['ltnet']],
    'DS - MACE - MV': [item[0] for item in list(big_samples_labels_map.items())
                       if item[1]['ds'] == item[1]['mace'] == item[1]['mv']],
    'DS - MACE - LTNet': [item[0] for item in list(big_samples_labels_map.items())
                          if item[1]['ds'] == item[1]['mace'] == item[1]['ltnet']],
    'DS - MV - LTNet': [item[0] for item in list(big_samples_labels_map.items())
                        if item[1]['mv'] == item[1]['ltnet'] == item[1]['ds']],
    'MACE - MV - LTNet': [item[0] for item in list(big_samples_labels_map.items())
                          if item[1]['mace'] == item[1]['mv'] == item[1]['ltnet']],
    'DS - MACE': [item[0] for item in list(big_samples_labels_map.items())
                  if item[1]['ds'] == item[1]['mace']],
    'DS - MV': [item[0] for item in list(big_samples_labels_map.items())
                if item[1]['ds'] == item[1]['mv']],
    'DS - LTNet': [item[0] for item in list(big_samples_labels_map.items())
                   if item[1]['ds'] == item[1]['ltnet']],
    'MACE - MV': [item[0] for item in list(big_samples_labels_map.items())
                  if item[1]['mace'] == item[1]['mv']],
    'MACE - LTNet': [item[0] for item in list(big_samples_labels_map.items())
                     if item[1]['mace'] == item[1]['ltnet']],
    'MV - LTNet': [item[0] for item in list(big_samples_labels_map.items())
                   if item[1]['mv'] == item[1]['ltnet']],
    'All different': [item[0] for item in list(big_samples_labels_map.items())
                      if item[1]['ds'] != item[1]['mace'] != item[1]['mv'] != item[1]['ltnet']],
    'Total': list(big_samples_labels_map.keys()),
}

methods = {
    'All match': ['ds', 'mace', 'mv', 'ltnet'],
    'DS - MACE - MV': ['ds', 'mace', 'mv'],
    'DS - MACE - LTNet': ['ds', 'mace', 'ltnet'],
    'DS - MV - LTNet': ['ds', 'mv', 'ltnet'],
    'MACE - MV - LTNet': ['mace', 'mv', 'ltnet'],
    'DS - MACE': ['ds', 'mace'],
    'DS - MV': ['ds', 'mv'],
    'DS - LTNet': ['ds', 'ltnet'],
    'MACE - MV': ['mace', 'mv'],
    'MACE - LTNet': ['mace', 'ltnet'],
    'MV - LTNet': ['mv', 'ltnet'],
    # 'All different': [],
}

alphas = {}
for key, sample_list in list(overlap.items()):
    alpha_entry = []
    for sample in sample_list:
        for approach, label in list(big_samples_labels_map[sample].items()):
            # if approach in methods[key]:
            alpha_entry.append((approach, sample, f'{label}'))
    if len(alpha_entry) != 0:
        alphas[key] = krippendorf_alpha(alpha_entry)

scores = {key: len(overlap[key]) for key in list(overlap.keys()) if len(overlap[key]) != 0}

# label_dist = {key: Counter([value[key] for value in list(big_samples_labels_map.values())])
#               for key in list(big_samples_labels_map.values())[0].keys()}
scores = dict(sorted(scores.items(), key=lambda item: item[1]))
alphas = {key: alphas[key] for key in list(scores.keys())}

# normalize scores
total_samples = len(set(big_samples_labels_map))
scores = {key: scores[key] / total_samples for key in list(scores.keys())}

# draw horizontal bars
# index = [method.replace('_', ' / ') if method.split('_')[0] != 'all'
#          else method.replace('_', ' ') for method in list(scores.keys())]
df = pd.DataFrame({'Samples (normalized)': list(scores.values()),
                   'Krippendorf\'s alpha': list(alphas.values())}, index=list(scores.keys()))
# fig = plt.figure(figsize=(10, 8))
plt.style.use('seaborn-whitegrid')
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"] = 1.25

ax = df.plot.barh(legend=False, color={'Samples (normalized)': colors[0], 'Krippendorf\'s alpha': colors[1]}, edgecolor='black', linewidth=2.0)
ax.legend(['Samples (normalized)', 'Krippendorf\'s alpha'], loc='best', bbox_to_anchor=(1.0, 0.12), fontsize=8)
# plt.title(f'Agreement of inferred labels for the {dataset_name.capitalize()} dataset')
# plt.ylabel('Methods')
# plt.xlabel('Samples')
save_path = f'{VISUALS_FOLDER}/{dataset_name}_inferred_labels_final.pdf'
if dataset_name == 'tripadvisor':
    if one_dataset_one_annotator is True:
        save_path = f'{VISUALS_FOLDER}/tripadvisor_task3_inferred_labels_final.pdf'
plt.savefig(save_path,
            bbox_inches='tight')
