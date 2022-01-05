import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from augment import get_data_transforms
from .utils import (
    SimpleDataset,
    InfiniteSampler,
)

# Logger for dataset script
import logging


__all__ = [
    'get_iters'
]


def get_iters(
        args,
        dataset = 'cifar10', 
        n_labelled = 4000, 
        n_valididation = 5000, 
        l_batch_size = 32, 
        ul_batch_size = 128, 
        test_batch_size = 256,
        data_transforms = None,
        pseudo_label = None,
        workers = 0, 
    ):

    # Load logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.info("Loading data")
    # Load datasets
    if dataset == 'cifar10':
        args.num_classes = 10
        # Get datasets from python with automatic downloading
        train_dataset = datasets.CIFAR10(root='./data', download=True, train=True, transform=None)
        test_dataset = datasets.CIFAR10(root='./data', download=True, train=False, transform=None)
        # Extract data for custom dataloaders
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
    elif dataset == 'cifar100':
        args.num_classes = 100
        # Get datasets from python with automatic downloading
        train_dataset = datasets.CIFAR100(root='./data', download=True, train=True, transform=None)
        test_dataset = datasets.CIFAR100(root='./data', download=True, train=False, transform=None)
        # Extract data for custom dataloaders
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
    elif dataset == 'svhn':
        args.num_classes = 10
        # Get datasets from python with automatic downloading
        train_dataset = datasets.SVHN(root='./data', download=True, split='train', transform=None)
        test_dataset = datasets.SVHN(root='./data', download=True, split='test', transform=None)
        # Extract data for custom dataloaders
        x_train, y_train = train_dataset.data, np.array(train_dataset.labels)
        x_test, y_test = test_dataset.data, np.array(test_dataset.labels)
        # Transpose data for compatibility
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    else: raise ValueError

    logger.info("Loading data transforms")
    # Load all needed data transforms
    if data_transforms is None:
        data_transforms = get_data_transforms(args)

    # Randomly permute data for split
    randperm = np.random.permutation(len(x_train))

    # Extract indices for labelled, validation, and unlabelled sets
    labelled_idx    = randperm[:n_labelled]
    validation_idx = randperm[n_labelled:n_labelled + n_valididation]
    unlabelled_idx  = randperm[n_labelled + n_valididation:]

    # Assign input data
    x_labelled    = x_train[labelled_idx]
    x_validation = x_train[validation_idx]
    x_unlabelled  = x_train[unlabelled_idx]

    # Assign output data
    y_labelled    = y_train[labelled_idx]
    y_validation = y_train[validation_idx]
    if pseudo_label is None:
        y_unlabelled = y_train[unlabelled_idx]
    else:
        assert isinstance(pseudo_label, np.ndarray)
        y_unlabelled = pseudo_label

    # If validation set has a size of zero, set it to testing set
    if n_valididation == 0: x_validation, y_validation = x_test, y_test

    data_iterators = {
        'labelled': DataLoader(
            SimpleDataset(x_labelled, y_labelled, transform_x = data_transforms['labelled']),
            batch_size = l_batch_size, num_workers = workers,
            sampler = InfiniteSampler(len(x_labelled)),
        ),
        'unlabelled': DataLoader(
            SimpleDataset(x_unlabelled, y_unlabelled, transform_x = data_transforms['unlabelled']),
            batch_size = ul_batch_size, num_workers = workers,
            sampler = InfiniteSampler(len(x_unlabelled)),
        ),
        'val': DataLoader(
            SimpleDataset(x_validation, y_validation, transform_x = data_transforms['test']),
            batch_size = test_batch_size, num_workers = workers, shuffle = False
        ),
        'test': DataLoader(
            SimpleDataset(x_test, y_test, transform_x = data_transforms['test']),
            batch_size = test_batch_size, num_workers = workers, shuffle = False
        )
    }

    # Logging dataset sizes
    logger.info("Number of labelled examples:   {}".format(str(len(x_labelled)).rjust(10)))
    logger.info("Number of unlabelled examples: {}".format(str(len(x_unlabelled)).rjust(10)))
    logger.info("Number of validation examples: {}".format(str(len(x_validation)).rjust(10)))
    logger.info("Number of test examples:       {}".format(str(len(x_test)).rjust(10)))

    return data_iterators