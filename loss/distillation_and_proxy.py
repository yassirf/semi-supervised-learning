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

__all__ = ['crossentropy_and_distillation_and_proxy']


def kl_divergence_loss(input_logits, target_logits, temperature):
    """
    Computes the temperature annealed kl-divergence
    Gradients only propagated through the input logits.
    """
    input_lsoftmax = F.log_softmax(input_logits/temperature, dim = -1)
    target_lsoftmax = F.log_softmax(target_logits/temperature, dim = -1)

    loss = torch.exp(target_lsoftmax) * (target_lsoftmax - input_lsoftmax)
    loss = (loss.sum(-1)).mean()
    return loss


def get_negative_log_confidence(target_logits):
    logc = torch.log_softmax(target_logits, dim = -1)
    logc = logc.max(dim = -1).values
    return -logc


def rank_confidence_loss(input_scalars, target_logits, param):
    """
    Computes a spearman rank approximated loss.
    Gradients only propagated through the input scalars.
    """

    # Get the target scalars (negate since we want uncertainty)
    target_scalars = get_negative_log_confidence(target_logits)

    # Compute the soft rank correlation score
    rank1 = soft_rank(input_scalars.unsqueeze(0), regularization_strength=param)
    rank2 = soft_rank(target_scalars.unsqueeze(0), regularization_strength=param)

    # Normalize and compute batch spearman
    rank1 = (rank1 - rank1.mean())/rank1.std(unbiased = False)
    rank2 = (rank2 - rank2.mean())/rank2.std(unbiased = False)

    spearman_loss = -(rank1 * rank2).mean()
    return spearman_loss


def rank_mae_loss(input_scalars, target_logits, param):

    # Get the target scalars (negate since we want uncertainty)
    target_scalars = get_negative_log_confidence(target_logits)

    # Compute max absolute error loss
    loss = torch.abs(input_scalars - target_scalars).mean()
    return loss


class DistillationProxy(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(DistillationProxy, self).__init__(args, model, optimiser, scheduler)

        # Get proxy loss weight and regularization strength for differentiable rank losses
        self.proxy_w = args.proxy_weight
        self.proxy_regularization_strength = args.proxy_regularization_strength

        # Proxy loss
        self.proxy_loss = rank_mae_loss

    def get_correlation_metrics(self, input_scalars, target_logits):

        target_logits = target_logits.detach().clone()
        input_scalars = input_scalars.detach().clone().cpu()

        # Get the target scalars (negate since we want uncertainty)
        target_scalars = get_negative_log_confidence(target_logits).cpu()

        # Compute correlations
        spear = scipy.stats.spearmanr(input_scalars, target_scalars)[0]
        pears = scipy.stats.pearsonr(input_scalars, target_scalars)[0]
        return spear, pears

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        pred_l, pred_info = self.model(x_l)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_l, _ = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_scalars = pred_info['proxy'], 
            target_logits = teacher_l, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_info['proxy'], teacher_l)

        # Compute accuracy
        acc = accuracy(pred_l.detach().clone(), y_l, top_k = (1, 5))

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'ce': ce.item(),
            'kd': kd.item(),
            'proxy': proxy.item(),
            'spear': spear,
            'pears': pears,
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}

        return loss, linfo



def crossentropy_and_distillation_and_proxy(**kwargs):
    return DistillationProxy(**kwargs)