import argparse

from torch._C import default_generator

__all__ = ['get_args']


def get_args():
    # Basic parameters
    parser = get_init_args()

    # Get loss args
    parser = get_loss_args(parser)

    # Get arch specific parameters
    parser = get_arch_spec_args(parser)

    # Get optimization args
    parser = get_optim_args(parser)

    # Get scheduler args
    parser = get_schedule_args(parser)

    # Get uncertainty args
    parser = get_uncertainty_args(parser)

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
    parser.add_argument('--distil-train-data', default=False, type=bool, help='whether training data should be used for distilation')

    parser.add_argument('--test-batch', default=100, type=int, help='test batchsize (default: 100)')
    parser.add_argument('--test-augment', default='none', type=str, help='test: nature of data augmentation (default: none)')

    parser.add_argument('--num-labelled',   default=4000, type=int, help='number of labelled examples (default: 4000)')
    parser.add_argument('--num-validation', default=5000, type=int, help='number of validation examples (default: 5000)')
    parser.add_argument('--num-unlabelled', default=0,    type=int, help='number of unlabelled examples (default: 0 = all)')

    # Architecture
    parser.add_argument('--arch', type=str, required=True, help='model architecture')

    # Checkpoints and saving
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--save-every', default=0, type=int, help='save every nth checkpoint (default: 0 — saving at the of training)')
    parser.add_argument('--log-every', default=1000, type=int, help='log after every nth update (default: 1000)')
    parser.add_argument('--val-every', default=None, type=int, help='validate after every nth update (default: None)')
    parser.add_argument('--load-path', default=None, type=str, help='path to load model (default: None)')
    parser.add_argument('--out-path', default=None, type=str, help='path to store results (default: None)')

    # Miscs
    parser.add_argument('--seed', default=None, type=int, help='manual seed for torch and numpy (default: None)')

    # Device options
    parser.add_argument('--gpu', default=1, type=int, help='Using CUDA for training (default: 1)')
    return parser


def get_loss_args(parser):
    # Loss arguments
    parser.add_argument('--loss', default='crossentropy', type=str, help='loss type (default: cross_entropy)')
    parser.add_argument('--teacher-path', default=None, type=str, help='path to teacher model in distillation losses (default: None)')
    parser.add_argument('--teacher-arch', default=None, type=str, help='teacher architecture in distillation losses (default: None)')
    parser.add_argument('--teacher-ratio', default=1.0, type=float, help='knowledge distillation weight (default: 1.0)')
    parser.add_argument('--temperature', default=1.0, type=float, help='knowledge distillation temperature (default: 1.0)')
    parser.add_argument('--proxy-weight', default=1.0, type=float, help='soft rank regularization strength (default: 1.0)')
    parser.add_argument('--proxy-regularization-strength', default=1.0, type=float, help='soft rank regularization strength (default: 1.0)')
    parser.add_argument('--vat-alpha', default=1.0, type=float, help='vat loss weighting (default: 1.0)')
    parser.add_argument('--vat-ent', default=1.0, type=float, help='vat entropy minimisation factor (default: 1.0)')
    parser.add_argument('--vat-xi', default=1e-6, type=float, help='vat small optimisation direction (default: 1e-8)')
    parser.add_argument('--vat-eps', default=8.0, type=float, help='vat optimisation size (default: 1.0)')
    parser.add_argument('--vat-ip', default=1, type=int, help='vat number of power iterations (default: 1)')
    parser.add_argument('--meanteacher-alpha-ramp', default=0.990, type=float, help='mean-teacher ema parameter during ramp-up (default: 0.990)')
    parser.add_argument('--meanteacher-alpha', default=0.999, type=float, help='mean-teacher ema parameter after ramp-up (default: 0.999)')
    parser.add_argument('--meanteacher-w', default=10.00, type=float, help='mean-teacher mixing loss coefficient (default: 10.00)')
    parser.add_argument('--meanteacher-i', default=15000, type=int, help='mean-teacher number of ramp-up iterations (default: 30000)')
    parser.add_argument('--mixup-alpha', default=0.0, type=float, help='mixup beta distribution parameters (default: 0.0)')
    parser.add_argument('--uce-mu', default=0.0, type=float, help='uce entropy regularisation constant (default: 0.0)')
    parser.add_argument('--uce-num-samples', default=1, type=int, help='uce monte-carlo samples at training (default: 1)')
    return parser


def get_arch_spec_args(parser):
    # Architecture specific parameters  latent_dim, flow_length
    parser.add_argument('--latent-dim', default=16, type=int, help='dimensionality of latent space (default: 16)')
    parser.add_argument('--flow-length', default=1, type=int, help='number of normalizing flow layers (default: 1)')
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


def get_uncertainty_args(parser):
    # Uncertainty and temperature scaling
    parser.add_argument('--uncertainty-name', default='categorical_ensemble', type=str, help='nature of uncertainty (default: categorical_ensemble)')
    parser.add_argument('--uncertainty-temperature', default=1.0, type=float, help='temperature scaling of logits (default: 1.0)')
    parser.add_argument('--uncertainty-samples', default=0, type=int, help='number of samples for intractable distributions (default: 0)')
    return parser


def process_args(parser):
    # Create args class
    args = parser.parse_args()
    return args