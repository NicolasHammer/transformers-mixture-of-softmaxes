import torch
from torch import Tensor
from torchtext.data.datasets_utils import _RawTextIterableDataset


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, train: _RawTextIterableDataset, valid: _RawTextIterableDataset, test: _RawTextIterableDataset):
        self.dictionary = Dictionary()
        self.train = self.tokenize(train, isTrain=True)
        self.valid = self.tokenize(valid)
        self.test = self.tokenize(test)

    def tokenize(self, raw_text_iter: _RawTextIterableDataset, isTrain=False) -> Tensor:
        if isTrain:
            data = []
            for line in raw_text_iter:
                words = ['<sos>'] + line.split() + ['<eos>']

                word2idx_list = []
                for word in words:
                    self.dictionary.add_word(word)
                    word2idx_list.append(self.dictionary.word2idx[word])
                data.append(torch.tensor(word2idx_list).type(torch.int64))
        else:
            data = []
            for line in raw_text_iter:
                words = ['<sos>'] + line.split() + ['<eos>']
                sent_tokenized = torch.tensor(
                    [self.dictionary.word2idx[word] for word in words]).type(torch.int64)
                data.append(sent_tokenized)
        
        return torch.cat(data)
