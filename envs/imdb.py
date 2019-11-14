import os
import torch
import torch.nn as nn
import torchtext
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

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
            tokenize=self.preprocess
        )
        self.label = torchtext.data.LabelField(dtype=torch.long)

        path = os.path.join(root, 'imdb', 'preprocessed.pth')
        if os.path.isfile(path):
            pth = torch.load(path, map_location=lambda storage, loc: storage)
            train = pth['train']
            test = pth['test']
        else:
            data = torchtext.datasets.IMDB
            self.pbar = tqdm(total=50000)
            train, test = data.splits(self.text, self.label, root=root)
            train = list(iter(train))
            test = list(iter(test))
            pth = {
                'train': train,
                'test': test
            }
            torch.save(pth, path)

        fields = [('text', self.text), ('label', self.label)]
        train = torchtext.data.Dataset(train, fields)
        self.train, self.val = train.split(split_ratio)
        self.test = torchtext.data.Dataset(test, fields)

    def build_vocab(self, vectors):
        self.text.build_vocab(self.train, vectors=vectors)
        self.label.build_vocab(self.train)
        return self.text.vocab.vectors

    def get_pos(self, token):
        tag = nltk.pos_tag([token])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text):
        text = nltk.word_tokenize(text.lower())
        text = [token for token in text if token.isalpha()]
        text = [self.lemmatizer.lemmatize(t, self.get_pos(t)) for t in text]
        text = [token for token in text if token not in self.stopwords]
        self.pbar.update()
        return text


def imdb_glove50d(args):
    root = os.path.join(PROJECT_ROOT, DATA_DIR)
    data = IMDB(root, args.split_ratio)
    vectors = torchtext.vocab.Vectors(
        'glove.6B.300d.txt',
        os.path.join(root, 'glove')
    )
    vocab = data.build_vocab(vectors)
    return {
        'size': (128, 300),
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
