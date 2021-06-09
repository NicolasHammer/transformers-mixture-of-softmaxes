import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class MixtureOfSoftmaxes(nn.Module):
    def __init__(self, num_softmaxes: int, ntoken: int, embedding_size: int, dropout: float):
        super(MixtureOfSoftmaxes, self).__init__()

        self.embedding_size = embedding_size
        self.ntoken = ntoken
        self.num_softmaxes = num_softmaxes

        self.prior = nn.Linear(embedding_size, num_softmaxes)
        self.latent = nn.Sequential(
            nn.Linear(embedding_size, num_softmaxes*embedding_size), nn.Tanh())
        self.decoder = nn.Linear(embedding_size, ntoken)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        latent = self.dropout(self.latent(input))
        logit = self.decoder(latent.view(-1, self.embedding_size))

        prior_logit = self.prior(input).view(-1, self.num_softmaxes)
        prior = F.softmax(prior_logit, -1)

        prob = F.softmax(logit.view(-1, self.ntoken), -
                         1).view(-1, self.num_softmaxes, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        output = torch.log(prob.add_(1e-8)).view(-1, self.ntoken)

        return output
