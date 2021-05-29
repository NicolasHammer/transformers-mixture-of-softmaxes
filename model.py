import math

import torch
from torch import Tensor
from torch import nn, functional as f


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, n_tokens, dim_model: int = 512):
        super().__init__()
        self.linear = nn.Linear(dim_model, n_tokens)

    def forward(self, x: Tensor):
        out = self.linear(x)
        return f.log_softmax(out, dim=-1)


class MosDecoder(nn.Module):
    def __init__(self, n_tokens, dim_model: int = 512):
        super().__init__()

    def forward(self, x: Tensor):
        pass


class Transformer(nn.Module):
    def __init__(
      self,
      n_tokens,
      decoder,
      num_encoder_layers: int = 6,
      dim_model: int = 512,
      num_heads: int = 8,
      n_ff_hidden_units: int = 2048,
      dropout: float = .1,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.embedding = nn.Embedding(n_tokens, dim_model)
        self.position_encoder = PositionalEncoding(dim_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads,
                                                   dim_feedforward=n_ff_hidden_units)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = decoder
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.position_encoder(src)
        return self.encoder(src, src_mask)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        h = self.encode(src, src_mask)
        return self.decoder(h)


def make_transformer(n_tokens, dim_model, n_heads, n_layers, n_ff_hid, dropout):
    decoder = Decoder(n_tokens, dim_model)
    return Transformer(
        n_tokens,
        decoder,
        n_layers,
        dim_model,
        n_heads,
        n_ff_hid,
        dropout
    )

def make__mos_transformer(n_tokens, dim_model, n_heads, n_layers, n_ff_hid, dropout):
    decoder = MosDecoder(n_tokens, dim_model)
    return Transformer(
        n_tokens,
        decoder,
        n_layers,
        dim_model,
        n_heads,
        n_ff_hid,
        dropout
    )
