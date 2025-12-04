import torch
from torch.distributions import RelaxedBernoulli, Beta


class ApproxBernoulli(RelaxedBernoulli):
    def rsample(self, sample_shape=torch.Size()):  # noqa
        x = super().rsample(sample_shape)
        return x - x.detach() + (x > 0.5).to(x.dtype)

def BetaPERT(m, lmd):
    alpha = 1 + lmd * m
    beta = 1 + lmd * (1-m)
    return Beta(alpha, beta)

def NLL_spk_cond(alpha, beta, u):
    """
    log likelihood of speaker conditional E_{q(\pi)}[\log p(u|\pi)]
    """
    return -(u * torch.digamma(alpha) + (1-u)*torch.digamma(beta) + torch.digamma(alpha+beta))