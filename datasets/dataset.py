import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from contextlib import contextmanager

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


# Setting local seeds
@contextmanager
def temporary_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


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
        seed = 0,
    ):

    # Load logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)
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
    elif dataset.startswith('cifar10-c-'):
        args.num_classes = 10
        # Corrupted cifar 10 name format: cifar10-c-i-type
        severity = int(dataset.split("-")[2])
        dsettype = dataset.split("-")[3]
        # Extract data for custom dataloaders
        x_train, y_train = np.load('./data/CIFAR-10-C/{}.npy'.format(dsettype)), np.load('./data/CIFAR-10-C/labels.npy').astype(int)
        x_test, y_test = x_train[(severity - 1) * 10000: severity * 10000], y_train[(severity - 1) * 10000: severity * 10000]
    elif dataset == 'cifar100':
        args.num_classes = 100
        # Get datasets from python with automatic downloading
        train_dataset = datasets.CIFAR100(root='./data', download=True, train=True, transform=None)
        test_dataset = datasets.CIFAR100(root='./data', download=True, train=False, transform=None)
        # Extract data for custom dataloaders
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
    elif dataset.startswith('cifar100-c-'):
        args.num_classes = 100
        # Corrupted cifar 100 name format: cifar10-c-i-type
        severity = int(dataset.split("-")[2])
        dsettype = dataset.split("-")[3]
        # Extract data for custom dataloaders
        x_train, y_train = np.load('./data/CIFAR-100-C/{}.npy'.format(dsettype)), np.load('./data/CIFAR-100-C/labels.npy').astype(int)
        x_test, y_test = x_train[(severity - 1) * 10000: severity * 10000], y_train[(severity - 1) * 10000: severity * 10000]
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

    # Setting local random number generator
    logger.info("Generating a dataset permutation with a local seed = {}".format(seed))

    with temporary_seed(seed):
        # Randomly permute data for split
        randperm = np.random.permutation(len(x_train))

    # Extract indices for labelled, validation, and unlabelled sets
    labelled_idx   = randperm[:n_labelled]
    validation_idx = randperm[n_labelled:n_labelled + n_valididation]
    unlabelled_idx = randperm[n_labelled + n_valididation:]

    # Assign input data
    x_labelled   = x_train[labelled_idx]
    x_validation = x_train[validation_idx]
    x_unlabelled = x_train[unlabelled_idx]

    # Assign output data
    y_labelled   = y_train[labelled_idx]
    y_validation = y_train[validation_idx]
    y_unlabelled = y_train[unlabelled_idx]

    # If validation set has a size of zero, set it to testing set
    if len(validation_idx) == 0: x_validation, y_validation = x_test, y_test

    # If unlabelled set had a size of zero, set it to train set
    if len(unlabelled_idx) == 0: x_unlabelled, y_unlabelled = x_train, y_train

    # Force the unlabelled dataset to utilise the full data
    # x_unlabelled, y_unlabelled = x_train, y_train

    # Logging dataset sizes
    logger.info("Number of labelled examples:   {}".format(str(len(x_labelled)).rjust(10)))
    logger.info("Number of unlabelled examples: {}".format(str(len(x_unlabelled)).rjust(10)))
    logger.info("Number of validation examples: {}".format(str(len(x_validation)).rjust(10)))
    logger.info("Number of test examples:       {}".format(str(len(x_test)).rjust(10)))
    
    data_iterators = {
        'labelled': iter(DataLoader(
            SimpleDataset(x_labelled, y_labelled, transform_x = data_transforms['labelled']),
            batch_size = l_batch_size, num_workers = workers,
            sampler = InfiniteSampler(len(x_labelled)),
        )),
        'unlabelled': iter(DataLoader(
            SimpleDataset(x_unlabelled, y_unlabelled, transform_x = data_transforms['unlabelled']),
            batch_size = ul_batch_size, num_workers = workers,
            sampler = InfiniteSampler(len(x_unlabelled)),
        )),
        'val': DataLoader(
            SimpleDataset(x_validation, y_validation, transform_x = data_transforms['test']),
            batch_size = test_batch_size, num_workers = workers, shuffle = False
        ),
        'test': DataLoader(
            SimpleDataset(x_test, y_test, transform_x = data_transforms['test']),
            batch_size = test_batch_size, num_workers = workers, shuffle = False
        )
    }

    return data_iterators