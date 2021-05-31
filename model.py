import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as f
from torch.autograd import Variable


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
    def __init__(self, num_softmaxes, ntoken, embedding_size,dropout):
        super(MosDecoder, self).__init__()

        self.embedding_size = embedding_size
        self.ntoken = ntoken
        self.num_softmaxes = num_softmaxes

        self.prior = nn.Linear(embedding_size, num_softmaxes, bias=False)
        self.latent = nn.Sequential(
            nn.Linear(embedding_size, num_softmaxes * embedding_size), nn.Tanh())
        self.decoder = nn.Linear(embedding_size, ntoken)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        latent = self.latent(input)
        latent = self.dropout(latent)
        logit = self.decoder(latent.view(-1, self.embedding_size))

        prior_logit = self.prior(input).view(-1, self.num_softmaxes)
        prior = f.softmax(prior_logit, -1)

        prob = f.softmax(logit.view(-1, self.ntoken), -
        1).view(-1, self.num_softmaxes, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        output = torch.log(prob.add_(1e-8)).view(-1, self.ntoken)

        return output


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

    def generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

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


def make_mos_transformer(n_experts, n_tokens, dim_model, n_heads, n_layers, n_ff_hid, dropout):
    decoder = MosDecoder(n_experts, n_tokens, dim_model,dropout)
    return Transformer(
        n_tokens,
        decoder,
        n_layers,
        dim_model,
        n_heads,
        n_ff_hid,
        dropout
    )
