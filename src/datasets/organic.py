from datasets import BaseDataset
"""
def file_processor(path, text_processor):
    data = pd.read_csv(path, sep='\t').rename(columns={"author_id": "author_name":"comment_nr":"sentence_nr":"domain_relevance":"sentiment":"entity":"attribute":"sentence":"source_file":"annotator"})
    print(data[0])
    data['embedding'] = None
    for index, row in data.iterrows():
        data.at[index, 'embedding'] = text_processor(row['text'])

    return data

def line_processor(line, text_processor):
    rating, text = (lambda x: (x[0], x[1]))(line.split('\t'))
    # one-hot encode ratings
    return {'label': one_hot_encode_ratings(rating), 'text': text, 'embedding': text_processor(text)}


def file_processor(path, text_processor, annotator):
    data = []
    f = open(path, 'r')
    for line in f:
        processed_line = line_processor(line, text_processor)
        if processed_line['label'] is not None:
            data.append({'annotator': annotator, **processed_line})

    return data


def one_hot_encode_ratings(rating):
    # set to None if label is to be excluded
    ratings_map = {
        '-4': 0,
        '-2': 0,
        '0': None,
        '2': 1,
        '4': 1,
    }
    if rating not in ratings_map.keys():
        print(f'Rating not in map: {rating}')
    return ratings_map[rating]


class TripAdvisorDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)

        self.size = size = args.get('size', 'max').lower()
        self.stars = stars = args.get('stars', 'All').lower().title()

        if size != 'max':
            if len(size) == 1:
                size += 'k'
            if size not in ['1k', '2k', '4k', '8k', '16k']:
                raise Exception('Size must be one of these: 1k, 2k, 4k, 8k, 16k or max')

        if stars != 'All':
            if len(stars) == 1:
                stars += '.0'
            if stars not in ['2.0', '3.0', '4.0']:
                raise Exception('Stars must be one of these: 2.0, 3.0, 4.0 or All')

        root = f'{self.root_data}tripadvisor/{size} text files'
        path_f = f'{root}/TripAdvisorUKHotels-{stars}-{size}_F.txt'
        path_m = f'{root}/TripAdvisorUKHotels-{stars}-{size}_M.txt'

        data_f = file_processor(path_f, self.text_processor, 'f')
        data_m = file_processor(path_m, self.text_processor, 'm')

        self.annotators = ['f', 'm']

        self.data = data_f + data_m

        self.data_shuffle()

# :"comment_nr":"sentence_nr":"domain_relevance":"sentiment":"entity":"attribute":"sentence":"source_file":"annotator"
"""
path = '../data/organic/processed_concatenated.csv'
data = pd.read_csv(path, sep='|').rename(columns={"author_id": "author_name":"comment_nr":"sentence_nr":"domain_relevance":"sentiment":"entity":"attribute":"sentence":"source_file":"annotator"})
print(data[0])
