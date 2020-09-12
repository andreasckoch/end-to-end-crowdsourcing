from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    """Dataset Template"""

    def __init__(self, **argv):
        self.argv = argv

        self.mode = 'train'
        self.train_val_split = argv.get('train_val_split', 0.8)

        self._build_text_processor(**argv)
        pass

    def _build_text_processor(self, **argv):
        text_processor = argv.get('text_processor', 'word2vec').lower()

        if text_processor == 'word2vec':
            from datasets.processors.word2vec import _build_text_processor, text_processor
            self.text_processor_model = _build_text_processor(**argv)
            self.text_processor_func = text_processor

    def text_processor(self, text, **argv):
        return self.text_processor_func(self.text_processor_model, text, **argv)

    def data_shuffle(self):
        import random 
        random.seed(123456789)
        random.shuffle(self.data)

        l = len(self.data)
        eof_train_split = int(l*self.train_val_split*0.9)
        eof_val_split = int(l*0.9)
        
        self.data = {
            'train': self.data[0:eof_train_split],
            'validation': self.data[eof_train_split:eof_val_split],
            'test': self.data[eof_val_split:]
        }

    def set_mode(self, mode):
        if mode not in ['train', 'validation', 'test']:
            raise Exception('mode must be train or validation or test')
        self.mode = mode

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):
        return self.data[self.mode][idx]