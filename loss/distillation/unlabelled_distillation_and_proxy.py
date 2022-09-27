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

from loss.base import accuracy
from loss.distillation.distillation_and_proxy import DistillationProxy
from loss.distillation.distillation_and_proxy import DistillationProxyEntropyMSE

import logging 
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = [
    'unlabelled_distillation_and_proxy_only',
    'unlabelled_distillation_and_proxy_entropy_mse_only',
]


class UnlabelledDistillationProxy(DistillationProxy):
    def __init__(self, args, model, optimiser, scheduler):
        super(UnlabelledDistillationProxy, self).__init__(args, model, optimiser, scheduler)

    def forward(self, info):

        # Get labelled image and label
        x_ul = info['x_ul']

        # Perform model forward pass and get the prediction info dictionary
        pred_ul, pred_info = self.model(x_ul)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_ul, _ = self.teacher(x_ul)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_ul, teacher_ul, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_scalars = torch.sigmoid(pred_info['proxy']), 
            target_logits = teacher_ul, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = kd * self.distillation_t ** 2 + proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(torch.sigmoid(pred_info['proxy']), teacher_ul)

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'kd': kd.item(),
            'proxy': proxy.item(),
            'spear': spear,
            'pears': pears,
        }}

        return loss, linfo


class UnlabelledDistillationProxyEntropyMSE(DistillationProxyEntropyMSE):
    def __init__(self, args, model, optimiser, scheduler):
        super(UnlabelledDistillationProxyEntropyMSE, self).__init__(args, model, optimiser, scheduler)

    def forward(self, info):

        # Get labelled image and label
        x_ul = info['x_ul']

        # Perform model forward pass and get the prediction info dictionary
        pred_ul, _ = self.model(x_ul)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_ul, _ = self.teacher(x_ul)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_ul, teacher_ul, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_logits = pred_ul, 
            target_logits = teacher_ul, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = kd * self.distillation_t ** 2 + proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_ul, teacher_ul)

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'kd': kd.item(),
            'proxy': proxy.item(),
            'spear': spear,
            'pears': pears,
        }}

        return loss, linfo


def unlabelled_distillation_and_proxy_only(**kwargs):
    return UnlabelledDistillationProxy(**kwargs)


def unlabelled_distillation_and_proxy_entropy_mse_only(**kwargs):
    return UnlabelledDistillationProxyEntropyMSE(**kwargs)
