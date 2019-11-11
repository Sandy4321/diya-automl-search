import os
import re
import torch
import torch.nn as nn
import torchtext
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from settings import PROJECT_ROOT, DATA_DIR
nltk.data.path.append(os.path.join(PROJECT_ROOT, DATA_DIR, 'nltk_data'))

__all__ = ['imdb_glove50d']


class DataLoader:
    def __init__(self, data, vocab, **kwargs):
        self.data = torchtext.data.BucketIterator(data, repeat=False, **kwargs)
        self.iter = iter(self.data)
        self.embed = nn.Embedding(vocab.size(0), vocab.size(1))
        self.embed.weight.data.copy_(vocab)

    def __iter__(self):
        self.iter = iter(self.data)
        return self

    def __next__(self):
        batch = next(self.iter)
        data = self.embed(batch.text)
        data = data.transpose(0, 1).detach()
        label = batch.label.detach()
        return data, label


class IMDB:
    def __init__(self, root, split_ratio, length=128):
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.text = torchtext.data.Field(
            lower=True,
            fix_length=length,
            tokenize=self.clean_text
        )
        self.label = torchtext.data.LabelField(dtype=torch.long)

        data = torchtext.datasets.IMDB
        train, self.test = data.splits(self.text, self.label, root=root)
        self.train, self.val = train.split(split_ratio)

    def build_vocab(self, vectors):
        self.text.build_vocab(self.train, vectors=vectors)
        self.label.build_vocab(self.train)
        return self.text.vocab.vectors

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
        text = text.lower()
        text = [self.lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [self.lemmatizer.lemmatize(token, "v") for token in text]
        text = [word for word in text if word not in self.stopwords]
        text = " ".join(text)
        return text


def imdb_glove50d(args):
    root = os.path.join(PROJECT_ROOT, DATA_DIR)
    data = IMDB(root, args.split_ratio)
    vectors = torchtext.vocab.Vectors(
        'glove.6B.50d.txt',
        os.path.join(root, 'glove')
    )
    vocab = data.build_vocab(vectors)
    return {
        'size': (128, 50),
        'num_classes': 2,
        'train': DataLoader(
            data.train,
            vocab,
            batch_size=args.batch_size,
        ),
        'val': DataLoader(
            data.val,
            vocab,
            batch_size=args.batch_size,
        ),
        'test': DataLoader(
            data.test,
            vocab,
            batch_size=args.batch_size,
        )
    }
