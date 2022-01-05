
from .input import (
    none_transform, 
    standard_transform
)

__all__ = [
    'get_data_transforms'
]


def get_data_transform(mode = 'none'):
    if mode in 'none_transform':
        return none_transform()
    if mode in 'standard_transform':
        return standard_transform()
    raise ValueError

def get_data_transforms(args):
    # Save all transforms into a single directory
    return {
        'labelled': get_data_transform(mode = args.train_l_augment),
        'unlabelled': get_data_transform(mode = args.train_ul_augment),
        'test': get_data_transform(mode = args.test_augment),
    }