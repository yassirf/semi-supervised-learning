import math
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

import fast_soft_sort
from fast_soft_sort import pytorch_ops
from fast_soft_sort.pytorch_ops import soft_rank, soft_sort

import scipy
import scipy.stats

from .base import accuracy
from .cross_entropy import CrossEntropy
from .distillation import Distillation

import logging 
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ['proxy_loss']


def get_entropy(target_logits):
    logp = torch.log_softmax(target_logits, dim = -1)
    entropy = -torch.exp(logp) * logp
    entropy = entropy.sum(-1)
    return entropy


def mean_absolute_error_loss(input_scalars, target_logits, param):

    # Get the target scalars (negate since we want uncertainty)
    target_scalars = get_entropy(target_logits)

    # Compute the mae loss
    return (torch.abs(target_scalars - input_scalars)).mean()


class DistillationProxy(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(DistillationProxy, self).__init__(args, model, optimiser, scheduler)

        # Get proxy loss weight and regularization strength for differentiable rank losses
        self.proxy_w = args.proxy_weight
        self.proxy_regularization_strength = args.proxy_regularization_strength

        # Proxy loss
        self.proxy_loss = mean_absolute_error_loss

    def get_correlation_metrics(self, input_scalars, target_logits):

        target_logits = target_logits.detach().clone()
        input_scalars = input_scalars.detach().clone().cpu()

        # Get the target scalars (negate since we want uncertainty)
        target_scalars = get_entropy(target_logits).cpu()

        # Compute correlations
        spear = scipy.stats.spearmanr(input_scalars, target_scalars)[0]
        pears = scipy.stats.pearsonr(input_scalars, target_scalars)[0]
        return spear, pears

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        _, pred_info = self.model(x_l)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_l, _ = self.teacher(x_l)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_scalars = pred_info['proxy'], 
            target_logits = teacher_l,
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_info['proxy'], teacher_l)

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'proxy': proxy.item(),
            'spear': spear,
            'pears': pears,
        }}

        return loss, linfo

    @torch.no_grad()
    def eval_forward(self, info):
        
        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        _, pred_info = self.model(x_l)

        # The second input is for the teacher model
        teacher_l, _ = self.teacher(x_l)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_scalars = pred_info['proxy'], 
            target_logits = teacher_l,
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_info['proxy'], teacher_l)

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'proxy': proxy.item(),
            'spear': spear,
            'pears': pears,
        }}

        return loss, linfo


def proxy_loss(**kwargs):
    return DistillationProxy(**kwargs)