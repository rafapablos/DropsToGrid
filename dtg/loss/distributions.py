from abc import ABC

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Distribution(nn.Module, ABC):
    n_params: int

    def compute_loss(self, y):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def std(self):
        raise NotImplementedError


# ---------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------


class ZeroInflatedGamma(Distribution):
    n_params = 3

    def __init__(self, x):
        super().__init__()
        x = rearrange(x, "b t h w c -> b t c h w")
        alphas_raw, betas_raw, logits_zero = x.chunk(3, dim=2)
        alphas = F.softplus(alphas_raw) + 1e-6
        betas = F.softplus(betas_raw) + 1e-6

        self.alphas = alphas
        self.betas = betas
        self.logits_zero = logits_zero

        self.gamma = td.Gamma(alphas, betas)
        self.bernoulli = td.Bernoulli(logits=logits_zero)

    def compute_loss(self, y):
        eps = 1e-6
        log_p_zero = -F.softplus(self.logits_zero)
        log_p_nonzero = -F.softplus(-self.logits_zero) + self.gamma.log_prob(y + eps)

        log_prob = torch.where(y <= 0.1, log_p_zero, log_p_nonzero)
        return log_prob.sum(dim=2)

    @property
    def mean(self):
        p = torch.sigmoid(self.logits_zero)  # p_nonzero
        mu = self.alphas / self.betas

        zero = torch.zeros_like(mu)

        return torch.where(p < 0.5, zero, mu)

    @property
    def std(self):
        p = torch.sigmoid(self.logits_zero)
        mu = self.alphas / self.betas
        var_gamma = self.alphas / (self.betas**2)
        var = (1 - p) * var_gamma + p * (1 - p) * mu**2
        return torch.sqrt(var + 1e-6)
