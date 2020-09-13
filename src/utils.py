import numpy as np
import nltk
import torch
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)


def get_word2vec_dict(filename):
    embeddings = {}
    with open(filename, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings


# convert words to vectors (use only lower case words as glove embedding matrix only accepts these)
def convert_to_vectors(tokenized_dataset, data_to_keep_synchronized, word_vectors, word_vectors_dim):
    data_vectors = []
    ignored_words = set([])
    empty_sentences_idx = []
    stop_words = set(stopwords.words('english'))
    for i, data in enumerate(tokenized_dataset, 0):
        vectors = []
        new_sentence = []
        sentence = data[0]
        for word in sentence:
            if word not in stop_words:
                try:
                    vectors.append(word_vectors[word.lower()])
                    new_sentence.append(word)
                except KeyError as e:
                    ignored_words.add(word)
            else:
                ignored_words.add(word)
        if len(vectors) == 0:
            empty_sentences_idx.append(i)
            continue
        data_vectors.append(vectors)

    # keep provided data field synchronized
    empty_sentences_idx.reverse()
    for i in empty_sentences_idx:
        del data_to_keep_synchronized[i]

    # padding
    max_length = max([len(v) for v in data_vectors])
    data_vectors = [np.array(v + (max_length - len(v)) * [word_vectors_dim * [0]]) for v in data_vectors]

    return data_vectors, data_to_keep_synchronized


class SimpleCustomBatch:
    """
    Create function to apply to batch to allow for memory pinning when using a custom batch/custom dataset.
    Following guide on https://pytorch.org/docs/master/data.html#single-and-multi-process-data-loading
    """

    def __init__(self, data, device):
        self.input = torch.stack([sample['text_tensor'] for sample in data]).to(device=device)
        self.target = torch.stack([sample['label'] for sample in data]).to(device=device)

    def pin_memory(self):
        self.input = self.input.pin_memory()
        self.target = self.target.pin_memory()
        return self


def collate_wrapper(batch, device=torch.device('cuda')):
    return SimpleCustomBatch(batch, device)
