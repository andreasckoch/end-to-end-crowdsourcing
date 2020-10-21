import torch

# https://huggingface.co/transformers/custom_datasets.html
# class CustomDataset has the same code as class IMDbDataset from the "Fine-Tuning with custom
# datasets" huggingface tutorial. (URL above) TODO: If we dont change it, cite huggingface.

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, annotators):
        self.encodings = encodings
        self.labels = labels
        self.annotators = annotators

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)