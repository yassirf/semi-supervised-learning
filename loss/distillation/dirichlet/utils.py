import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence


__all__ = [
    'dirichlet_kl_divergence',
    'DirichletEstimation',
]


def dirichlet_kl_divergence(log_alphas, log_alphas_target, temperature_scale_num):

    # Get target scaled distributions
    alphas_target = torch.exp(log_alphas_target / temperature_scale_num)
    alphas_target = Dirichlet(alphas_target)

    # Get prediction scaled distribution
    alphas = torch.exp(log_alphas / temperature_scale_num)
    alphas = Dirichlet(alphas)

    # Use built in kl divergence (batch)
    loss = kl_divergence(alphas_target, alphas)

    return loss.mean()


class DirichletEstimation(object):
    def __init__(
            self,
            logprobs: torch.Tensor,
            temperature_scale: float = 2.0,
            estimation_iter: int = 5,
            beta = 1.0
    ):
        self.logprobs = logprobs.clone().detach()/temperature_scale
        self.estimation_iter = estimation_iter
        self.eps_init = 1e-3
        self.eps_step = 1e-6

        # For the softplus operation
        self.beta = beta

        # Logprobs should have size (batch, models, classes)
        assert logprobs.dim() == 3

    @torch.no_grad()
    def estimation_init(self):
        """
        Initialises the mean and scale of the estimated dirichlet.
        """
        # Normalise log probabilities
        self.logprobs = torch.log_softmax(self.logprobs, dim = -1)

        # Extract size
        b, m, v = self.logprobs.size()

        # Compute all necessary quantities
        log_e_prob = torch.logsumexp(self.logprobs, dim=1) - np.log(m)
        log_e_sq_prob = torch.logsumexp(2 * self.logprobs, dim=1) - np.log(m)
        log_e_prob_sq = 2 * log_e_prob

        # Used to initialise alpha0
        alpha0 = torch.exp(log_e_prob - log_e_sq_prob) - 1
        alpha0 = alpha0 / (1 - torch.exp(log_e_prob_sq - log_e_sq_prob) + self.eps_init)

        # Average over all estimates in log space
        alpha0 = torch.exp(torch.mean(torch.log(alpha0), dim = -1, keepdim = True))

        info = {
            'expected_log_prob': self.logprobs.mean(dim = 1),
            'log_expected_prob': log_e_prob,
            'expected_prob': torch.exp(log_e_prob),
            'alpha0': alpha0
        }
        return info

    @torch.no_grad()
    def estimation_step(self, info):
        """
        Performs an update step of the scale, alpha0
        """
        expected_log_prob = info['expected_log_prob']
        log_expected_prob = info['log_expected_prob']
        expected_prob  = info['expected_prob']
        initial_alpha0 = info['alpha0']

        # Number of classes size
        v = expected_prob.size(-1)

        # Sequence of steps to update alpha0
        new_alpha0 = torch.digamma(initial_alpha0 * expected_prob) - expected_log_prob
        new_alpha0 = (new_alpha0 * expected_prob).sum(dim=-1, keepdim=True)

        # Following steps are numerically instable
        new_alpha0 += (v - 1) / (initial_alpha0 + self.eps_step) - torch.digamma(initial_alpha0 + self.eps_step)
        new_alpha0 = (v - 1) / (new_alpha0 + self.eps_step)  # Adding errors to ensure it works

        info['alpha0'] = F.softplus(new_alpha0, beta=self.beta)
        return info

    @torch.no_grad()
    def __call__(self):
        # Initialise quantities
        info = self.estimation_init()

        # Perform estimation
        for i in range(self.estimation_iter):
            info = self.estimation_step(info)

        alpha0 = info['alpha0']

        # Due to highly confident predictions, overflows need to be removed
        mask = torch.isnan(alpha0)

        # Replace all nans with the mean of the rest
        alpha0[mask] = alpha0[~mask].mean()

        # Simply enforce all log alphas to be positive to avoid inverted dirichlet target
        estimated_log_alphas = F.softplus(info['log_expected_prob'] + torch.log(alpha0), beta=self.beta)
        return estimated_log_alphas