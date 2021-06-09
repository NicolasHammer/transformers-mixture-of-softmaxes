import torch
from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.vocab import Vocab


def data_process(raw_text_iter: _RawTextIterableDataset, vocab: Vocab, tokenizer) -> torch.Tensor:
    data = [torch.tensor([vocab[token] for token in tokenizer(item)],
                         dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
