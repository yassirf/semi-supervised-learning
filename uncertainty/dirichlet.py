import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Dict
from .categorical import EnsembleCategoricals

__all__ = [
    'ensemble_dirichlets'
]


class EnsembleDirichlets(EnsembleCategoricals):
    def __init__(self):
        super(EnsembleDirichlets, self).__init__()

    def compute_expected_entropy(self, log_alphas):
        alphas = torch.exp(log_alphas)
        alpha0 = torch.sum(alphas, dim=-1)

        entropy = torch.digamma(alpha0 + 1)
        entropy -= torch.sum(alphas * torch.digamma(alphas + 1), dim=-1) / alpha0

        return entropy.mean(0)

    @staticmethod
    def compute_reverse_mutual_information(expected, log_alphas):
        alphas = torch.exp(log_alphas)
        alpha0 = torch.sum(alphas, dim=-1)

        entropy = torch.exp(expected) * (expected - torch.digamma(alphas).mean(0))
        entropy = entropy.sum(-1) + torch.digamma(alpha0).mean(0)
        return entropy

    @torch.no_grad()
    def __call__(self, args, info: Dict, labels: Tensor, key: str = 'pred') -> Dict:
        """
        Computes all default uncertainty metrics
        """

        # Combine all outputs into a single tensor (samples, batch, classes)
        outputs = info[key]

        # Input dimension
        n, batch, classes = outputs.size()

        # Normalise results (samples, batch, classes)
        outputs_lp = torch.log_softmax(outputs, dim=-1)

        # Expected results (batch, classes)
        expected = torch.logsumexp(outputs_lp, dim=0) - np.log(n)

        returns = {
            'misclassification': self.compute_misclassification(expected, labels),
            'disagreement': self.compute_disagreement(expected),
            'log_confidence': -self.compute_log_confidence(expected),
            'entropy_expected': self.compute_entropy_expected(expected),
            'expected_entropy': self.compute_expected_entropy(outputs),
        }
        returns['mutual_information'] = returns['entropy_expected'] - returns['expected_entropy']
        returns['mutual_information'] = returns['entropy_expected'] - returns['expected_entropy']
        returns['reverse_mutual_information'] = self.compute_reverse_mutual_information(expected, outputs)
        returns['expected_pairwise_kl'] = returns['mutual_information'] + returns['reverse_mutual_information']
        return returns


def ensemble_dirichlets():
    return EnsembleDirichlets()