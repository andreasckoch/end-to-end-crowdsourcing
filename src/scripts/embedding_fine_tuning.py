"""
Script for fine tuning a glove embedding adapted from 
https://towardsdatascience.com/fine-tune-glove-embeddings-using-mittens-89b5f3fe4c39
"""

import csv
import pickle
import numpy as np
from collections import Counter
from nltk.corpus import brown
from mittens import GloVe, Mittens
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

from datasets.emotion import EmotionDataset
from datasets.organic import OrganicDataset


def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                 for line in reader}
    return embed


glove_path = "../data/embeddings/word2vec/glove.6B.50d.txt"  # get it from https://nlp.stanford.edu/projects/glove
pre_glove = glove2dict(glove_path)

# take words from our own dataset
# label_dim = 3
# annotator_dim = 38
# dataset = EmotionDataset()
# dataset_name = 'emotion'
# emotion = 'valence'
# dataset.set_emotion(emotion)

label_dim = 3
annotator_dim = 10
padding_length = 136
predict_coarse_attributes_task = False
dataset = OrganicDataset(predict_coarse_attributes_task=predict_coarse_attributes_task,
                         padding_length=padding_length)
dataset_name = 'organic'


sw = list(stop_words.ENGLISH_STOP_WORDS)
# dataset_unique_words = brown.words()[:200000]
dataset_unique_words = [word for sample in dataset for word in sample['text'].split()]
dataset_nonstop = [token.lower() for token in dataset_unique_words if (token.lower() not in sw)]
# oov = [token for token in dataset_nonstop if token not in pre_glove.keys()]

dataset_doc = [' '.join(dataset_nonstop)]

cv = CountVectorizer(ngram_range=(1, 1))
X = cv.fit_transform(dataset_doc)
corp_vocab = cv.vocabulary_.keys()  # list(set(oov))
Xc = (X.T * X)
Xc.setdiag(0)
coocc_ar = Xc.toarray()

mittens_model = Mittens(n=50, max_iter=1000)

new_embeddings = mittens_model.fit(
    coocc_ar,
    vocab=corp_vocab,
    initial_embedding_dict=pre_glove)

newglove = dict(zip(corp_vocab, new_embeddings))
f = open(f"../data/embeddings/word2vec/fine_tuned/{dataset_name}_glove.pkl", "wb")
pickle.dump(newglove, f)
f.close()
