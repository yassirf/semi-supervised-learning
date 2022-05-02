import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Dict
from .categorical import EnsembleCategoricals

__all__ = [
    'ensemble_gaussian_logits'
]


class EnsembleGaussianLogits(EnsembleCategoricals):
    def __init__(self):
        super(EnsembleGaussianLogits, self).__init__()

    @torch.no_grad()
    def __call__(self, args, info: Dict, labels: Tensor, key: str = 'pred') -> Dict:

        # Get the samples from info
        return super(EnsembleGaussianLogits, self).__call__(
            args, info, labels, key = 'samples'
        )


def ensemble_gaussian_logits():
    return EnsembleGaussianLogits()