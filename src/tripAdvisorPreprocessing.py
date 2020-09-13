from utils import get_word2vec_dict, convert_to_vectors

import numpy as np
import random
import torch
from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer

DATA_PATH_MALE = '../data/tripAdvisor/max text files/TripAdvisorUKHotels-All-max_M.txt'
DATA_PATH_FEMALE = '../data/tripAdvisor/max text files/TripAdvisorUKHotels-All-max_F.txt'
GLOVE_PATH = '../data/embeddings/glove.6B.50d.txt'


class tripAdvisorDataset(Dataset):

    def __init__(self):
        self.word_vectors = get_word2vec_dict(GLOVE_PATH)
        self.word_vectors_dim = 50
        self.fraction_of_dataset = 0.1
        self.train_val_split = 0.8
        self.tok = RegexpTokenizer(r'\w+')
        self.random_seed = 12345678
        self.mode = 'train'
        self.gender = 'male'
        self.device = torch.device('cuda')

        male_dataset, female_dataset = [], []
        with open(DATA_PATH_MALE, 'r') as file:
            for i, line in enumerate(file):
                label, text = line.rstrip().split('\t', 1)
                label = np.array(label).astype(np.float32)
                male_dataset.append([text, label])
        with open(DATA_PATH_FEMALE, 'r') as file:
            for i, line in enumerate(file):
                label, text = line.rstrip().split('\t', 1)
                label = np.array(label).astype(np.float32)
                female_dataset.append([text, label])

        # shuffle data and split it into train/val sets
        random.seed(self.random_seed)
        random.shuffle(male_dataset)
        random.shuffle(female_dataset)
        male_idx_split = int(self.fraction_of_dataset * self.train_val_split * len(male_dataset))
        female_idx_split = int(self.fraction_of_dataset * self.train_val_split * len(female_dataset))
        male_idx_end = int(self.fraction_of_dataset * len(male_dataset))
        female_idx_end = int(self.fraction_of_dataset * len(female_dataset))
        self.train_male_data = male_dataset[:male_idx_split]
        self.val_male_data = male_dataset[male_idx_split:male_idx_end]
        self.train_female_data = female_dataset[:female_idx_split]
        self.val_female_data = female_dataset[female_idx_split:female_idx_end]

        # tokenize
        train_male_tokenized_dataset = [[self.tok.tokenize(x[0]), x[1]] for x in self.train_male_data]
        val_male_tokenized_dataset = [[self.tok.tokenize(x[0]), x[1]] for x in self.val_male_data]
        train_female_tokenized_dataset = [[self.tok.tokenize(x[0]), x[1]] for x in self.train_female_data]
        val_female_tokenized_dataset = [[self.tok.tokenize(x[0]), x[1]] for x in self.val_female_data]

        print('Done tokenizing!')

        # convert to vectors
        self.train_male_vectors, self.train_male_data = convert_to_vectors(
            train_male_tokenized_dataset, self.train_male_data, self.word_vectors, self.word_vectors_dim)
        self.val_male_vectors, self.val_male_data = convert_to_vectors(
            val_male_tokenized_dataset, self.val_male_data, self.word_vectors, self.word_vectors_dim)
        self.train_female_vectors, self.train_female_data = convert_to_vectors(
            train_female_tokenized_dataset, self.train_female_data, self.word_vectors, self.word_vectors_dim)
        self.val_female_vectors, self.val_female_data = convert_to_vectors(
            val_female_tokenized_dataset, self.val_female_data, self.word_vectors, self.word_vectors_dim)

        print('Finished preprocessing!')

    def __len__(self):
        if self.gender == 'male' and self.mode == 'train':
            return len(self.train_male_data)
        elif self.gender == 'male' and self.mode == 'val':
            return len(self.val_male_data)
        elif self.gender == 'male' and self.mode == 'full':
            return len(self.train_male_data) + len(self.val_male_data)
        elif self.gender == 'female' and self.mode == 'train':
            return len(self.train_female_data)
        elif self.gender == 'female' and self.mode == 'val':
            return len(self.val_female_data)
        elif self.gender == 'female' and self.mode == 'full':
            return len(self.train_female_data) + len(self.val_female_data)

    def train_mode(self):
        self.mode = 'train'

    def val_mode(self):
        self.mode = 'val'

    def full_mode(self):
        self.mode = 'full'

    def male(self):
        self.gender = 'male'

    def female(self):
        self.gender = 'female'

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        datapoint = None
        label = None
        text = None
        if self.gender == 'male' and self.mode == 'train':
            datapoint = self.train_male_vectors[index]
            text = self.train_male_data[index][0]
            label = self.train_male_data[index][1]
        elif self.gender == 'male' and self.mode == 'val':
            datapoint = self.val_male_vectors[index]
            text = self.val_male_data[index][0]
            label = self.val_male_data[index][1]
        elif self.gender == 'male' and self.mode == 'full':
            datapoint = self.train_male_vectors[index] if index < len(self.train_male_data) else self.val_vectors[index - len(self.train_male_data)]
            text = self.train_male_data[index][0] if index < len(self.train_male_data) else self.val_vectors[index - len(self.train_male_data)][0]
            label = self.train_male_data[index][1] if index < len(self.train_male_data) else self.val_vectors[index - len(self.train_male_data)][1]
        elif self.gender == 'female' and self.mode == 'train':
            datapoint = self.train_female_vectors[index]
            text = self.train_female_data[index][0]
            label = self.train_female_data[index][1]
        elif self.gender == 'female' and self.mode == 'val':
            datapoint = self.val_female_vectors[index]
            text = self.val_female_data[index][0]
            label = self.val_female_data[index][1]
        elif self.gender == 'female' and self.mode == 'full':
            datapoint = self.train_female_vectors[index] if index < len(
                self.train_female_data) else self.val_vectors[index - len(self.train_female_data)]
            text = self.train_female_data[index][0] if index < len(
                self.train_female_data) else self.val_vectors[index - len(self.train_female_data)][0]
            label = self.train_female_data[index][1] if index < len(
                self.train_female_data) else self.val_vectors[index - len(self.train_female_data)][1]

        return {
            'label': torch.tensor(label, device=self.device, dtype=torch.float32),
            'text': text,
            'text_tensor': torch.tensor(datapoint, device=self.device, dtype=torch.float32)}
