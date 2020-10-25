from datasets import BaseDataset


def line_processor(line, text_processor):
    words = line.split(',')
    
    return {
        'id': words[0], 
        'sentence': words[1],
        'template': words[2],
        'person': words[3],
        'gender': words[4],
        'race': words[5],
        'emotion': words[6],
        'emption_word': words[7],
        'embedding': text_processor(words[1])
    }


def file_processor(path, text_processor, annotator):
    data, annotators = [], []
    f = open(path, 'r')
    for line in f:
        processed_line = line_processor(line, text_processor)
        annotators.append(processed_line['person'])
        if processed_line['label'] is not None:
            data.append({'annotator': processed_line['person'], 'pseudo_labels': {}, **processed_line})

    return data, list(set(annotators))


class EquityDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)

        root = f'{self.root_data}equity/Equity-Evaluation-Corpus.csv'

        self.data, self.annotators = file_processor(root, self.text_processor)

        self.data_shuffle()
