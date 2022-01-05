import random
import numpy as np
import torch

__all__ = ['set_seed']


def set_seed(args):

    if args.seed is None:
        args.seed = random.randint(1, 10000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)