import argparse

from torch._C import default_generator

__all__ = ['get_args']


def get_args():
    # Basic parameters
    parser = get_init_args()

    # Get loss args
    parser = get_loss_args(parser)

    # Get optimization args
    parser = get_optim_args(parser)

    # Get scheduler argparse
    parser = get_schedule_args(parser)

    # Format arguments
    return process_args(parser)


def get_init_args():
    parser = argparse.ArgumentParser(description = 'PyTorch Image Classification Training')

    # Datasets
    parser.add_argument('--dataset', default='cifar10', type=str, help='image classification dataset (default: cifar10)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')

    # Training options
    parser.add_argument('--iters', default=10000, type=int, help='number of iterations to train (default: 10000)')
    parser.add_argument('--num-augments', default=2, type=int, help='number of repeated augmentations in dedicated augment options (default: 2)')
    parser.add_argument('--randaugment-n', default=2, type=int, help='randaugment number of augmentation (default: 2)')
    parser.add_argument('--randaugment-m', default=9, type=int, help='randaugment number of possible augmentations (default: 9)')

    parser.add_argument('--train-l-batch', default=128, type=int, help='train labelled batchsize (default: 128)')
    parser.add_argument('--train-l-augment', default='standard', type=str, help='labelled: nature of data augmentation (default: standard)')

    parser.add_argument('--train-ul-batch', default=128, type=int, help='train unlabelled batchsize (default: 128)')
    parser.add_argument('--train-ul-augment', default='standard', type=str, help='unlabelled: nature of data augmentation (default: standard)')
    
    parser.add_argument('--test-batch', default=100, type=int, help='test batchsize (default: 100)')
    parser.add_argument('--test-augment', default='none', type=str, help='test: nature of data augmentation (default: none)')

    parser.add_argument('--num-labelled', default=4000, type=int, help='number of labelled examples (default: 4000)')
    parser.add_argument('--num-validation', default=5000, type=int, help='number of validation examples (default: 5000)')

    # Architecture
    parser.add_argument('--arch', type=str, required=True, help='model architecture')

    # Checkpoints and saving
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--save-every', default=0, type=int, help='save every nth checkpoint (default: 0 — saving at the of training)')
    parser.add_argument('--log-every', default=1000, type=int, help='log after every nth update (default: 1000)')
    parser.add_argument('--val-every', default=None, type=int, help='validate after every nth update (default: None)')

    # Miscs
    parser.add_argument('--seed', default=None, type=int, help='manual seed for torch and numpy (default: None)')

    # Device options
    parser.add_argument('--gpu', default=1, type=int, help='Using CUDA for training (default: 1)')
    return parser


def get_loss_args(parser):
    # Loss arguments
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss type (default: cross_entropy)')
    parser.add_argument('--vat_alpha', default=1.0, type=float, help='vat loss weighting (default: 1.0)')
    parser.add_argument('--vat_ent', default=1.0, type=float, help='vat entropy minimisation factor (default: 1.0)')
    parser.add_argument('--vat_xi', default=1e-6, type=float, help='vat small optimisation direction (default: 1e-8)')
    parser.add_argument('--vat_eps', default=8.0, type=float, help='vat optimisation size (default: 1.0)')
    parser.add_argument('--vat_ip', default=1, type=int, help='vat number of power iterations (default: 1)')
    parser.add_argument('--meanteacher_alpha_ramp', default=0.990, type=float, help='mean-teacher ema parameter during ramp-up (default: 0.990)')
    parser.add_argument('--meanteacher_alpha', default=0.999, type=float, help='mean-teacher ema parameter after ramp-up (default: 0.999)')
    parser.add_argument('--meanteacher_w', default=10.00, type=float, help='mean-teacher mixing loss coefficient (default: 10.00)')
    parser.add_argument('--meanteacher_i', default=15000, type=int, help='mean-teacher number of ramp-up iterations (default: 30000)')
    return parser


def get_optim_args(parser):
    # Optimiser and learning rate
    parser.add_argument('--optim', default='sgd', type=str, help='optimiser type (default: sgd)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--learning-rate-final', default=0.0, type=float, help='final learning rate (default: 0.0 — not always used)')
    return parser


def get_schedule_args(parser):
    # Scheduler and learning rate
    parser.add_argument('--lr-scheduler', default='multistep', type=str, help='type of learning rate schedule (default: multistep)')
    parser.add_argument('--milestones', default=[0.3, 0.6, 0.8], type=float,  nargs='+', help='multisteplr fractional milestones (default: [0.3, 0.6, 0.8])')
    parser.add_argument('--gamma', default=0.2, type=float, help='multisteplr multiplicative decay (default: 0.2)')
    return parser


def process_args(parser):
    # Create args class
    args = parser.parse_args()
    return args