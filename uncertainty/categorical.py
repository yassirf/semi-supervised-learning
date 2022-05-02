import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from .base import BaseClass

__all__ = [
    'ensemble_categoricals'
]


class EnsembleCategoricals(BaseClass):
    def __init__(self):
        super(EnsembleCategoricals, self).__init__()

    @staticmethod
    def compute_disagreement(log_probs):
        return 1 - torch.exp(2 * log_probs).sum(-1)

    @staticmethod
    def compute_log_confidence(log_probs):
        return log_probs.max(axis=-1).values

    @staticmethod
    def compute_entropy(log_probs):
        entropy = - log_probs * torch.exp(log_probs)
        return entropy.sum(-1)

    def compute_expected_entropy(self, log_probs):
        # Input of shape (samples, batch, classes)
        entropies = self.compute_entropy(log_probs)
        return entropies.mean(0)

    def compute_entropy_expected(self, log_probs):
        return self.compute_entropy(log_probs)

    @staticmethod
    def compute_reverse_mutual_information(expected, log_probs):
        entropy = expected - log_probs.mean(0)
        entropy = (torch.exp(expected) * entropy).sum(-1)
        return entropy

    @torch.no_grad()
    def __call__(self, args, info: Dict, key: str = 'pred') -> Dict:
        """
        Computes all default uncertainty metrics
        """

        # Assert temperature parameter exists
        temperature = getattr(args, "uncertainty_temperature")

        # Combine all outputs into a single tensor (samples, batch, classes)
        outputs = info[key]

        # Input dimension
        n, batch, classes = outputs.size()

        # Create the zero matrix
        zero = torch.zeros_like(outputs[0, :, 0])

        # Temperature anneal
        outputs = outputs / temperature

        # Normalise results (samples, batch, classes)
        outputs = torch.log_softmax(outputs, dim=-1)

        # Expected results (batch, classes)
        expected = torch.logsumexp(outputs, dim=0) - np.log(n)

        returns = {
            'disagreement': self.compute_disagreement(expected),
            'log_confidence': -self.compute_log_confidence(expected),
            'entropy_expected': self.compute_entropy_expected(expected),
            'expected_entropy': self.compute_expected_entropy(outputs) if n > 1 else zero,
        }
        returns['mutual_information'] = returns['entropy_expected'] - returns['expected_entropy'] if n > 1 else zero
        returns['reverse_mutual_information'] = self.compute_reverse_mutual_information(expected, outputs) if n > 1 else zero
        returns['expected_pairwise_kl'] = returns['mutual_information'] + returns['reverse_mutual_information'] if n > 1 else zero
        return returns


def ensemble_categoricals():
    return EnsembleCategoricals()