"""
Generic training script for both supervised and semi-supervised models.
Yassir Fathullah 2022
"""

import copy
import time
import torch
import torch.nn as nn

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
    best_loss, best_top1, best_top5 = 0.0, 0.0, 0.0

    # Checkpointer for saving
    checkpointer = utils.Checkpointer(path = args.checkpoint, save_last_n = -1)

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
            msg  = 'iteration: {}\tlr: {:.5f}\t'.format(str(i).rjust(6), loss.lr)
            for key, value in loss.metrics.items():
                msg += '{}: {:.3f}\t'.format(key, value.avg)
            logger.info(msg)
            loss.reset_metrics()

        if i % (args.val_every or args.log_every) == 0:

            # Certain methods have better performance with a teacher like mean-teacher
            # Note all loss functions should have a "get_validation_model" method
            valmodel = loss.get_validation_model()

            # Run validation set with a copied model
            val_loss, val_top1, val_top5 = test(args, logger, device, data_iterators['val'], copy.deepcopy(valmodel).to(device))

            # Save model checkpoint
            checkpointer.save(i, val_top1, model, loss, optimiser)

            # Update best metrics
            if val_top1 > best_top1:
                best_loss = val_loss
                best_top1 = val_top1
                best_top5 = val_top5

            # Logging validation 
            msg = "test opt\tloss: {loss:.3f} | top1: {top1: .3f} | top5: {top5: .3f}"
            msg = msg.format(loss=best_loss, top1=best_top1, top5=best_top5)
            logger.info(msg)
    
    # Save last model 
    checkpointer.save(i, 0.0, model, loss, optimiser)


@torch.no_grad()
def test(args, logger, device, dataloader, model):

    # Set model into evaluation mode
    model.eval()

    # Evaluation trackers
    criterion = nn.CrossEntropyLoss()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    t0 = time.time()
    # Single testing loop
    for x, y in dataloader:

        # Move to cuda if being used
        x, y = x.to(device), y.to(device)

        # Make prediction with model
        pred, _ = model(x)

        # And get metrics
        loss = criterion(pred, y)
        accs = accuracy(pred.detach().clone(), y, top_k = (1, 5))

        # Update trackers
        losses.update(loss.data.item(), x.size(0))
        top1.update(accs[0].item(), x.size(0))
        top5.update(accs[1].item(), x.size(0))

    # Logging message
    msg = "test\tloss: {loss:.3f} | top1: {top1: .3f} | top5: {top5: .3f} | time {time: .3f}"
    msg = msg.format(loss=losses.avg, top1=top1.avg, top5=top5.avg, time=time.time()-t0)
    logger.info(msg)

    return losses.avg, top1.avg, top5.avg


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

    # Train and test model
    logger.info("Training phase")
    train(args, logger, device, data_iters, model, optimiser, scheduler, loss)

    logger.info("Evaluation phase")
    test(args, logger, device, data_iters['test'], model)


if __name__ == '__main__':
    main()