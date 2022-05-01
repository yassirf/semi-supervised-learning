"""
Generic testing script for both supervised and semi-supervised models.
Yassir Fathullah 2022
"""

import json
import copy
import time
from regex import P
import torch
import torch.nn as nn
from typing import Dict

import datasets
import utils
from utils.meter import AverageMeter
from uncertainty.utils import UncertaintyStorage
from loss.base import accuracy

# Logger for main training script
import logging


def reshaper(pred, info):

    # Prediction to be modified as this is used in uncertainty computations
    ipred = info['pred']

    # Expand the info pred if necessary
    info['pred'] = ipred if ipred.dim() == 3 else ipred.unsqueeze(0)

    return pred, info


@torch.no_grad()
def test(args, logger, device, dataloader, model, uncertainty):

    # Set model into evaluation mode
    model.eval()

    # Calculate and store all uncertainties
    uncertainty_storage = UncertaintyStorage()

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
        pred, info = model(x)

        # Reshape into correct format
        pred, info = reshaper(pred, info)

        # And get metrics
        loss = criterion(pred, y)
        accs = accuracy(pred.detach().clone(), y, top_k = (1, 5))

        # Update trackers
        losses.update(loss.data.item(), x.size(0))
        top1.update(accs[0].item(), x.size(0))
        top5.update(accs[1].item(), x.size(0))

        # Calculate uncertainties
        results = uncertainty(args, info)

        import pdb; pdb.set_trace()
        
        uncertainty_storage.push(results)

    # Logging message
    msg = "test\tloss: {loss:.3f} | top1: {top1: .3f} | top5: {top5: .3f} | time {time: .3f}"
    msg = msg.format(loss=losses.avg, top1=top1.avg, top5=top5.avg, time=time.time()-t0)
    logger.info(msg)

    # Combine results
    results = {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
        'uncertainties': uncertainty_storage,
    }

    return results


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

    # Checkpointer for loading
    checkpointer = utils.Checkpointer(args, path = args.checkpoint, save_last_n = -1)
    model.load_state_dict(checkpointer.load(device)['state_dict'])

    # Load uncertainty calculating class
    uncertainty = utils.loaders.load_uncertainty(args)

    # Test model and generate predictions
    logger.info("Evaluation phase")
    results = test(args, logger, device, data_iters['test'], model, uncertainty)

    import pdb; pdb.set_trace()

    # Serializing json 
    json_object = json.dumps(results, indent = 4)
    
    # Writing to file
    with open(args.out_path, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main()