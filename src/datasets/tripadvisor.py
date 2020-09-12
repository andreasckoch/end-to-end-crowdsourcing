from datasets import BaseDataset


def line_processor(line, text_processor):
    rating, text = (lambda x: (x[0], x[1]))(line.split('\t'))
    return {'rating': rating, 'text': text, 'embedding': text_processor(text)}


def file_processor(path, text_processor, label):
    data = []
    f = open(path, 'r')
    for line in f:
        data.append({'label': label, **line_processor(line, text_processor)})

    return data


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


        root = f'../data/tripadvisor/{size} text files'
        path_f = f'{root}/TripAdvisorUKHotels-{stars}-{size}_F.txt'
        path_m = f'{root}/TripAdvisorUKHotels-{stars}-{size}_M.txt'

        data_f = file_processor(path_f, self.text_processor, 'f')
        data_m = file_processor(path_m, self.text_processor, 'm')

        self.data = data_f + data_m
        
        self.data_shuffle()
