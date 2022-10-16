"""
Generic training script for both supervised and semi-supervised models.
Yassir Fathullah 2022
"""

import copy
import time
import torch
import torch.nn as nn
import wandb

import datasets
import utils
from utils.meter import AverageMeter
from loss.base import accuracy

# Logger for main training script
import logging


def train(args, logger, device, data_iterators, model, optimiser, scheduler, loss):

    # Set model into training mode
    model.train()

    # Best loss and accuracy tracker
    best_metric_tracker = -1.0
    best_eval_metric = None

    # Checkpointer for saving
    checkpointer = utils.Checkpointer(args, path = args.checkpoint, save_last_n = -1)

    # Single training loop measured by number of batches
    for i in range(1, args.iters + 1):
       
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

            # Log in wandb
            train_metrics = {key: value.avg for key, value in loss.metrics.items()}
            wandb.log(train_metrics)

            msg  = 'iteration: {}\tlr: {:.5f}\t'.format(str(i).rjust(6), loss.lr)
            for key, value in loss.metrics.items():
                msg += '{}: {:.3f}\t'.format(key, value.avg)

            logger.info(msg)
            loss.reset_metrics()

        if i % (args.val_every or args.log_every) == 0:

            # Certain methods have better performance with a teacher like mean-teacher
            # Note all loss functions should have a "get_validation_model" method and discard any non graph leaves of the model
            valmodel = utils.loaders.load_model(args).to(device)
            valmodel.load_state_dict(loss.get_validation_model().state_dict())

            # Run validation set with a copied model
            eval_metrics = test(args, logger, device, data_iterators['val'], valmodel, loss)

            # Best metric tracker
            metric_tracker = eval_metrics['val-acc1'] if not args.track_spear else eval_metrics['val-spear']
            if metric_tracker > best_metric_tracker:
                best_metric_tracker = metric_tracker
                best_eval_metric = eval_metrics

            # Save model checkpoint
            checkpointer.save(i, metric_tracker, model, loss, optimiser)

            # Log in wandb
            wandb.log(eval_metrics)

            # Log in standard output
            msg = 'test     ||| '
            for key, value in eval_metrics.items():
                msg += '{}: {:.3f}\t'.format(key, value)
            logger.info(msg)

            msg = 'test opt ||| '
            for key, value in best_eval_metric.items():
                msg += '{}: {:.3f}\t'.format(key, value)
            logger.info(msg)

    # Save last model 
    checkpointer.save(i, 0.0, model, loss, optimiser)


@torch.no_grad()
def test(args, logger, device, dataloader, model, loss):

    # Set model into evaluation mode
    model.eval()

    t0 = time.time()
    # Single testing loop
    for x, y in dataloader:

        # Initialise input to loss and move to device
        loss_input_info = {}
        loss_input_info['x_l'] = x.to(device)
        loss_input_info['y_l'] = y.to(device)

        # Make prediction with model
        loss(loss_input_info, valmodel = model, batch_size = x.size(0), evaluation = True)

    eval_metrics = loss.eval_metrics
    eval_metrics = {key: value.avg for key, value in eval_metrics.items()}
    loss.reset_eval_metrics()
    return eval_metrics


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
        n_unlabelled = args.num_unlabelled,
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
    logger.info("Number of parameters: {:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1.0e6))

    # Load optimiser
    logger.info("Creating optimiser: {}".format(args.optim))
    optimiser = utils.loaders.load_optim(args, model)

    # Load scheduler
    logger.info("Creating learning rate scheduler: {}".format(args.lr_scheduler))
    scheduler = utils.loaders.load_schedule(args, optimiser)

    # Load loss
    logger.info("Creating loss: {}".format(args.loss))
    loss = utils.loaders.load_loss(args, model, optimiser, scheduler)

    # Setup wandb
    utils.setup_wandb(args, model)
    
    # Train and test model
    logger.info("Training phase")
    train(args, logger, device, data_iters, model, optimiser, scheduler, loss)

    logger.info("Evaluation phase")
    test(args, logger, device, data_iters['test'], model, loss)


if __name__ == '__main__':
    main()