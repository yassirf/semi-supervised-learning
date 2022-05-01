import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from .categorical import EnsembleCategoricals

__all__ = [
    'ensemble_laplace_logits'
]


class EnsembleLaplaceLogits(EnsembleCategoricals):
    def __init__(self):
        super(EnsembleLaplaceLogits, self).__init__()

    @torch.no_grad()
    def __call__(self, args, info: Dict, key: str = 'pred') -> Dict:

        # Get the samples from info
        return super(EnsembleLaplaceLogits, self).__call__(
            args, info, key = 'samples'
        )


def ensemble_laplace_logits():
    return EnsembleLaplaceLogits()
