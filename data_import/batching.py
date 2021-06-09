from typing import Tuple
from torch import Tensor
from torch.types import Device

def batchify(data: Tensor, bsz: int, device: Device) -> Tensor:
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, chunk_length: int) -> Tuple[Tensor, Tensor]:
    seq_len = min(chunk_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target