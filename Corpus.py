import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, device):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), isTrain=True).to(device)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt')).to(device)
        self.test = self.tokenize(os.path.join(path, 'test.txt')).to(device)

    def tokenize(self, path, isTrain=False):
        assert os.path.exists(path)

        if isTrain:
            with open(path, 'r', encoding="utf8") as file:
                for line in file:
                    words = ['<sos>'] + line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)

        with open(path, 'r', encoding="utf8") as f:
            data = []
            for line in f:
                words = ['<sos>'] + line.split() + ['<eos>']
                sent_tokenized = torch.tensor([self.dictionary.word2idx[word] for word in words]).type(torch.int64)
                data.append(sent_tokenized)
            return torch.cat(data)