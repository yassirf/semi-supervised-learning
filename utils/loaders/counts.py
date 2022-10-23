import torch

__all__ = ['load_counts']


def get_device(gpu = True):
    # Check if device is wanted and available
    available = gpu and torch.cuda.is_available()

    return torch.device('cuda' if available else 'cpu')


# Get all cifar models available in the folder given they are lower case and callable 
names = {
    'cifar10': [5000] * 10,
    'cifar100': [500] * 100,
    'svhn': [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]
}


# Define model loader
def load_counts(args):
    """
    General count loader
    """
    device = get_device(gpu = args.gpu)

    if args.dataset.startswith("cifar10-c-"):
        counts = names['cifar10']
    elif args.dataset.startswith("cifar100-c-"):
        counts = names['cifar100']
    elif args.dataset in names:
        counts = names[args.dataset]
    else:
        raise ValueError("The value {} is not one of the valid choices: {}".format(args.dataset, list(names.keys())))
    
    return torch.tensor(counts).to(device)