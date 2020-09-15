import numpy as np
from nltk.tokenize import RegexpTokenizer


def _build_text_processor(**argv):
    lang = argv.get('lang', 'en')
    tokenizer = RegexpTokenizer(r'\w+')
    padding_length = int(argv.get('padding_length', 100))
    embedding_dim = int(argv.get('embedding_dim', 50))

    embedding_path = argv.get('embedding_path', '../data/embeddings/word2vec/glove.6B.50d.txt')
    embeddings = {}
    with open(embedding_path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings, tokenizer, padding_length, embedding_dim


def text_processor(model, line, **argv):
    embeddings, tokenizer, padding_length, embedding_dim = model

    tokenized = tokenizer.tokenize(line)
    vectors = []
    for word in tokenized:
        try:
            vectors.append(embeddings[word])
        except KeyError as e:
            pass

    # cut vectors if it is longer than padding_length
    if len(vectors) > padding_length:
        vectors = vectors[:padding_length]

    # pad vectors to padding_length
    try:
        vectors = np.array(vectors + (padding_length - len(vectors)) * [embedding_dim * [0]])
    except TypeError as te:
        print(f'padding_length: {padding_length}, embedding_dim: {embedding_dim}\nVectors: {vectors}')
        import sys
        sys.exit()
    return vectors
