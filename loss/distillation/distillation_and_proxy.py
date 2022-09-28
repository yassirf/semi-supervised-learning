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
from loss.cross_entropy import CrossEntropy
from loss.distillation.distillation import Distillation

import logging 
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = [
    'crossentropy_and_distillation_and_proxy',
    'crossentropy_and_distillation_and_proxy_entropy_mse',
]


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


def get_normalized_entropy(target_logits):
    k = target_logits.size(-1)
    logp = torch.log_softmax(target_logits, dim = -1)
    entropy = -torch.exp(logp) * logp
    entropy = entropy.sum(-1)
    return entropy/math.log(k)


def binary_cross_entropy_loss(input_scalars, target_logits, param):

    loss = nn.BCELoss()

    # Get the target scalars (negate since we want uncertainty)
    target_scalars = get_normalized_entropy(target_logits)

    # Compute the binary cross entropy loss
    return loss(input_scalars, target_scalars)


def get_entropy(logits):
    logp = torch.log_softmax(logits, dim = -1)
    entropy = -torch.exp(logp) * logp
    entropy = entropy.sum(-1)
    return entropy


def mean_squared_error_loss(input_logits, target_logits, param):

    # Get the target scalars (negate since we want uncertainty)
    target_scalars = get_entropy(target_logits)
    input_scalars = get_entropy(input_logits)

    # Compute the binary cross entropy loss
    # return ((target_scalars - input_scalars) ** 2).mean()
    return (torch.abs(target_scalars - input_scalars)).mean()


class DistillationProxy(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(DistillationProxy, self).__init__(args, model, optimiser, scheduler)

        # Get proxy loss weight and regularization strength for differentiable rank losses
        self.proxy_w = args.proxy_weight
        self.proxy_regularization_strength = args.proxy_regularization_strength

        # Proxy loss
        self.proxy_loss = binary_cross_entropy_loss

    def get_correlation_metrics(self, input_scalars, target_logits):

        target_logits = target_logits.detach().clone()
        input_scalars = input_scalars.detach().clone().cpu()

        # Get the target scalars (negate since we want uncertainty)
        target_scalars = get_normalized_entropy(target_logits).cpu()

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
            input_scalars = torch.sigmoid(pred_info['proxy']), 
            target_logits = teacher_l, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(torch.sigmoid(pred_info['proxy']), teacher_l)

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

    @torch.no_grad()
    def eval_forward(self, info):
        
        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        pred_l, pred_info = self.valmodel(x_l)

        # The second input is for the teacher model
        teacher_l, _ = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_scalars = torch.sigmoid(pred_info['proxy']), 
            target_logits = teacher_l, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(torch.sigmoid(pred_info['proxy']), teacher_l)

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


class DistillationProxyEntropyMSE(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(DistillationProxyEntropyMSE, self).__init__(args, model, optimiser, scheduler)

        # Get proxy loss weight and regularization strength for differentiable rank losses
        self.proxy_w = args.proxy_weight
        self.proxy_regularization_strength = args.proxy_regularization_strength

        # Proxy loss
        self.proxy_loss = mean_squared_error_loss

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        pred_l, _ = self.model(x_l)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_l, _ = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_logits = pred_l, 
            target_logits = teacher_l, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_l, teacher_l)

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

    @torch.no_grad()
    def eval_forward(self, info):
        
        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        pred_l, _ = self.valmodel(x_l)

        # The second input is for the teacher model
        teacher_l, _ = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_logits = pred_l, 
            target_logits = teacher_l, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_l, teacher_l)

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


def crossentropy_and_distillation_and_proxy_entropy_mse(**kwargs):
    return DistillationProxyEntropyMSE(**kwargs)
