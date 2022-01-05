import numpy as np
import torch

__all__ = ['cosine']


def cosine(args, optimiser):
    
    # Total number of iterations
    total = args.iters

    # Multiplicative function
    lr_lambda = lambda i: np.cos(np.pi/2 * 7/8 * i/total)

    # Scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda = lr_lambda)