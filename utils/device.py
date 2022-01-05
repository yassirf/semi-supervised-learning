import torch

__all__ = ['get_device']


def get_device(gpu = True):
    # Check if device is wanted and available
    available = gpu and torch.cuda.is_available()

    return torch.device('cuda' if available else 'cpu')