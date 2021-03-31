from datasets import BaseDataset


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
            data.append({'annotator': annotator, 'pseudo_labels': {}, **processed_line})

    return data


def one_hot_encode_ratings(rating):
    # set to None if label is to be excluded
    ratings_map = {
        '-4': 0,
        '-2': 0,
        '0': 2,
        '2': 1,
        '4': 1,
    }
    if rating not in ratings_map.keys():
        print(f'Rating not in map: {rating}')
    return ratings_map[rating]


def add_noise(data, percent):
    import random
    random.seed(123456789)

    noised = []
    for item in data:
        r = random.uniform(0, 1)
        if r >= percent:
            noised.append({'noise': False, **item})
        else:
            noised.append({'noise': True, **item,
                           'original_label': item['label'],
                           'label': random.randint(0, 1)
                           })
    return noised


class TripAdvisorDataset(BaseDataset):
    def __init__(self, **args):
        super().__init__(**args)

        self.size = size = args.get('size', 'max').lower()
        self.stars = stars = args.get('stars', 'All').lower().title()
        self.male_noise = male_noise = args.get('male_noise', 0)
        self.female_noise = female_noise = args.get('female_noise', 0)

        self.one_dataset_one_annotator = args.get('one_dataset_one_annotator', False)

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

        if self.one_dataset_one_annotator is False:
            path_f = f'{root}/TripAdvisorUKHotels-{stars}-{size}_F.txt'
            path_m = f'{root}/TripAdvisorUKHotels-{stars}-{size}_M.txt'

            data_f = file_processor(path_f, self.text_processor, 'f')
            data_m = file_processor(path_m, self.text_processor, 'm')

            self.annotators = ['f', 'm']

            if male_noise != 0:
                data_m = add_noise(data_m, male_noise)

            if female_noise != 0:
                data_f = add_noise(data_f, female_noise)

            self.data = data_f + data_m
        else:
            path_hotels = f'{root}/TripAdvisorUKHotels-{stars}-{size}_MF.txt'
            path_restaurants = f'{root}/TripAdvisorUKRestaurant-{size}_MF.txt'
            self.annotators = ['hotels', 'restaurants']

            data_hotels = file_processor(path_hotels, self.text_processor, self.annotators[0])
            data_restaurants = file_processor(path_restaurants, self.text_processor, self.annotators[1])

            self.data = data_hotels + data_restaurants

        no_shuffle = args.get('no_shuffle', False)
        if no_shuffle is False:
            self.data_shuffle()
