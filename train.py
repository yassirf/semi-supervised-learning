"""
Generic training script for both supervised and semi-supervised models.
Yassir Fathullah 2022
"""

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import torch
import utils
from utils.meter import AverageMeter
import datasets
from loss import crossentropy

# Logger for main training script
import logging


def train(args, logger, device, data_iterators, model, optimiser, scheduler, loss):

    # Set model into training mode
    model.train()

    # Best accuracy tracker
    best_accuracy = 0.0

    # Checkpointer for saving
    checkpointer = utils.Checkpointer(path = args.checkpoint, save_last_n = -1)

    # Single training loop measured by number of batches
    for i in range(args.iters):
        
        # Reset loss trackers
        if i % args.log_every == 0:
            loss.reset_metrics()
        
        # Get labelled and unlabelled examples
        x_l, y_l = next(data_iterators['labelled'])
        x_ul, _ = next(data_iterators['unlabelled'])

        # Initialise input to loss and move to device
        loss_input_info = {}
        loss_input_info['x_l'] = x_l.to(device)
        loss_input_info['y_l'] = y_l.to(device)
        loss_input_info['x_ul'] = x_ul.to(device)

        # Perform a forward/backward pass and update learning rate
        loss(loss_input_info)

        if i % args.log_every == 0:

            msg  = 'iteration: {}\t'.format(str(i).rjust(6))
            msg += 'lr: {:.5f}\t'.format(loss.lr)
            for key, value in loss.metrics.items():
                msg += '{}: {:.3f}\t'.format(key, value.avg)
            logger.info(msg)

        if i > 0 and i % (args.val_every or args.log_every) == 0:
            # Run validation set
            # acc1 = test(args, logger, device, data_iterators['val'], model.clone())

            acc1 = AverageMeter()

            # Update best accuracy
            best_accuracy = max(acc1.avg, best_accuracy)

            # Validation log
            logger.info("test\tbest acc1: {:.3f}".format(best_accuracy))

            # Save models
            checkpointer.save(i, acc1.avg, model, optimiser)
            pass
    
    # Save last model 
    checkpointer.save(i, None, model, optimiser)


@torch.no_grad()
def test(args, logger, device, dataloader, model):
    
    # Create the cross-entropy loss for tracking
    loss = crossentropy(args = args, model = model, optimiser = None, scheduler = None)
    loss.reset_metrics()

    # Set model into evaluation mode
    model.eval()

    # Single testing loop
    for i, (x, y) in enumerate(dataloader):

        # Move to cuda if being used
        x, y = x.to(device), y.to(device)

        # Initialise input to loss and move to device
        loss_input_info = {}
        loss_input_info['x_l'] = x.to(device)
        loss_input_info['y_l'] = y.to(device)

        # Compute losses and metrics
        loss.eval_forward(loss_input_info, batch_size = x.size(0))

    msg = 'test\t'
    for key, value in loss.metrics.items():
        msg += '{}: {:.3f}\t'.format(key, value.avg)
    logger.info(msg)

    return loss.metrics['acc1']


def main():

    # Load logger
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Load arguments
    logger.info("Loading arguments")
    args = utils.get_args()
    # logger.info(args)

    # Set device
    device = utils.get_device(gpu = args.gpu)
    logger.info("Setting device: {}".format(device))

    # Set seeds
    utils.set_seed(args)
    logger.info("Setting seed: {}".format(args.seed))

    # Load datasets
    logger.info("Loading datasets")
    data_iters = datasets.get_iters(
        args, 
        dataset = args.dataset,
        n_labelled = args.num_labelled,
        n_valididation = args.num_validation,
        l_batch_size = args.train_l_batch, 
        ul_batch_size = args.train_ul_batch, 
        test_batch_size = args.test_batch, 
        data_transforms = None,
        pseudo_label = None,
        workers = args.workers
    )

    # Load model
    logger.info("Creating model: {}".format(args.arch))
    model = utils.loaders.load_model(args).to(device)
    logger.info("Number of parameters: {:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Load optimiser
    logger.info("Creating optimiser: {}".format(args.optim))
    optimiser = utils.loaders.load_optim(args, model)

    # Load scheduler
    logger.info("Creating learning rate scheduler: {}".format(args.lr_scheduler))
    scheduler = utils.loaders.load_schedule(args, optimiser)

    # Load loss
    logger.info("Creating loss: {}".format(args.loss))
    loss = utils.loaders.load_loss(args, model, optimiser, scheduler)

    # Train and test model
    logger.info("Training phase")
    train(args, logger, device, data_iters, model, optimiser, scheduler, loss)

    logger.info("Evaluation phase")
    test(args, logger, device, data_iters, model)


if __name__ == '__main__':
    main()