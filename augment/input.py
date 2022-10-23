import torch
import torchvision.transforms as transforms
from RandAugment import RandAugment

__all__ = [
    'none_transform',
    'standard_transform',
    'randaugment_transform',
    'split_standard_transform',
    'split_randaugment_transform',
    'composed_standard_transform',
    'composed_randaugment_transform',
]


def none_transform(args):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def standard_transform(args):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def split_standard_transform(args):
    # Get the standard transformer
    transform = standard_transform(args)

    # Define a stochastic augmenter which stacks images
    def repeated(x, num = args.num_augments):
        return torch.stack([x] + [transform(x) for _ in range(num - 1)])
    
    # Return the augmentation with stacking mechanism 
    return repeated


def composed_standard_transform(args):
    # Get the standard transformer
    transform = standard_transform(args)

    # Define a stochastic augmenter which stacks images
    def repeated(x, num = args.num_augments):
        return torch.stack([transform(x) for _ in range(num)])
    
    # Return the augmentation with stacking mechanism 
    return repeated


def randaugment_transform(args):

    # Get RandAugment hyperparameters
    n = args.randaugment_n
    m = args.randaugment_m

    return transforms.Compose([
        transforms.ToPILImage(),
        RandAugment(n, m),
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def split_randaugment_transform(args):
    # Get the standard transformer
    transform = randaugment_transform(args)

    # Define a stochastic augmenter which stacks images
    def repeated(x, num = args.num_augments):
        return torch.stack([x] + [transform(x) for _ in range(num - 1)])
    
    # Return the augmentation with stacking mechanism 
    return repeated


def composed_randaugment_transform(args):
    # Get the standard transformer
    transform = randaugment_transform(args)

    # Define a stochastic augmenter which stacks images
    def repeated(x, num = args.num_augments):
        return torch.stack([transform(x) for _ in range(num)])
    
    # Return the augmentation with stacking mechanism 
    return repeated
