
from .input import (
    none_transform, 
    standard_transform,
    randaugment_transform,
    split_standard_transform,
    split_randaugment_transform,
    composed_standard_transform,
    composed_randaugment_transform,
)

__all__ = [
    'get_data_transforms'
]


def get_data_transform(args, mode = 'none'):
    if mode in 'none_transform':
        return none_transform(args)
    if mode in 'standard_transform':
        return standard_transform(args)
    if mode in 'randaugment_transform':
        return randaugment_transform(args)
    if mode in 'split_standard_transform':
        return split_standard_transform(args)
    if mode in 'split_randaugment_transform':
        return split_randaugment_transform(args)
    if mode in 'composed_standard_transform':
        return composed_standard_transform(args)
    if mode in 'composed_randaugment_transform':
        return composed_randaugment_transform(args)
    raise ValueError


def get_data_transforms(args):
    # Save all transforms into a single directory
    return {
        'labelled': get_data_transform(args, mode = args.train_l_augment),
        'unlabelled': get_data_transform(args, mode = args.train_ul_augment),
        'test': get_data_transform(args, mode = args.test_augment),
    }