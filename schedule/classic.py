import numpy as np
import torch

__all__ = ['cosine', 'cosine1', 'cosine2', 'cosine3', 'cosinept']


def cosine(args, optimiser):
    
    # Total number of iterations
    total = args.iters

    # Multiplicative function
    lr_lambda = lambda i: np.cos(np.pi/2 * 7/8 * i/total)

    # Scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda = lr_lambda)


def cosine1(args, optimiser):
    
    # Total number of iterations
    total = args.iters

    # Multiplicative function
    lr_lambda = lambda i: (1 + np.cos(np.pi * i/total)) / 2

    # Scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda = lr_lambda)


def cosine2(args, optimiser):
    
    # Total number of iterations
    total = args.iters

    # Multiplicative function
    lr_lambda = lambda i: ((1 + np.cos(np.pi * i/total)) / 2) ** 2.0

    # Scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda = lr_lambda)


def cosine3(args, optimiser):
    
    # Total number of iterations
    total = args.iters

    # Multiplicative function
    lr_lambda = lambda i: (1 + np.cos(np.pi * i/total)) / 2 * 0.999 + 0.001
    
    # Scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda = lr_lambda)


def cosinept(args, optimiser):
    
    # Total number of iterations
    total = args.iters

    # Starting and ending learning rates
    lr_start = args.learning_rate
    lr_end = args.learning_rate_final

    ratio = lr_end/lr_start

    # Multiplicative function
    lr_lambda = lambda i: ratio + 0.50 * (1.0 - ratio) * (1 + np.cos(np.pi * i/total))
    
    # Scheduler
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda = lr_lambda)