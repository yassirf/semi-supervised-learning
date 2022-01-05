import argparse

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
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

    # Training options
    parser.add_argument('--iters', default=10000, type=int, help='number of iterations to train (default: 10000)')
    parser.add_argument('--train-l-batch', default=128, type=int, help='train labelled batchsize (default: 128)')
    parser.add_argument('--train-l-augment', default='standard', type=str, help='labelled: nature of data augmentation (default: standard)')

    parser.add_argument('--train-ul-batch', default=512, type=int, help='train unlabelled batchsize (default: 128)')
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
    return parser


def get_optim_args(parser):
    # Optimiser and learning rate
    parser.add_argument('--optim', default='sgd', type=str, help='optimiser type (default: sgd)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    return parser


def get_schedule_args(parser):
    # Scheduler
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='type of learning rate schedule (default: cosine)')
    return parser


def process_args(parser):
    # Create args class
    args = parser.parse_args()
    return args