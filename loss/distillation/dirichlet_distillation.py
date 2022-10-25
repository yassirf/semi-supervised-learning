import math
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

import torchsort
from torchsort import soft_rank, soft_sort

import scipy
import scipy.stats

from loss.base import accuracy
from loss.cross_entropy import CrossEntropy
from loss.distillation.distillation import Distillation
from models.cifar.ensemble import Ensemble

from .dirichlet import dirichlet_kl_divergence
from .dirichlet import DirichletEstimation


import logging 
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = [
    'crossentropy_and_dirichlet_distillation',
]


def dirichlet_estimation_loss(input_logits, target_logits, temperature):
    """
    Computes the temperature annealed dirichlet kl-divergence
    Gradients only propagated through the input logits.
    """
    
    # Build the estimator 
    estimator = DirichletEstimation(target_logits.permute(1, 0, 2))

    # Get the new true log_alphas
    target_logits = estimator()

    # Get the loss based on dirichlet kl
    loss = dirichlet_kl_divergence(
        log_alphas = input_logits, 
        log_alphas_target = target_logits, 
        temperature_scale_num = temperature,
    )

    return loss


def get_entropy(logits):
    logp = torch.log_softmax(logits, dim = -1)
    entropy = -torch.exp(logp) * logp
    entropy = entropy.sum(-1)
    return entropy


def get_mutual_information(logits, all_logits):
    # logits have shape (batch, classes)
    entropy = get_entropy(logits)

    # all_logits have shape (members, batch, classes)
    average_entropy = get_entropy(all_logits).mean(0)

    # Mutual information is the difference
    return entropy - average_entropy


def get_dirichlet_mutual_information(logits):
    # Entropy of expected
    entropy = get_entropy(logits)

    # Get the alphas and compute the expected entropy
    alphas = torch.exp(logits)
    alpha0 = torch.sum(alphas, dim=-1)

    ex_entropy = torch.digamma(alpha0 + 1)
    ex_entropy -= torch.sum(alphas * torch.digamma(alphas + 1), dim=-1) / alpha0
    return entropy - ex_entropy


class DirichletDistillation(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(DirichletDistillation, self).__init__(args, model, optimiser, scheduler)

        # Get proxy weighting loss
        self.proxy_w = args.proxy_weight

        # Proxy dirichlet loss
        self.proxy_loss = dirichlet_estimation_loss

    def build_teacher(self):
        # Get device to build model on
        device = utils.get_device(gpu = self.args.gpu)

        # Log the teacher information
        logger.info("Building teacher; architecture = {}".format(self.args.teacher_arch))
        logger.info("Building teacher; path = {}".format(self.args.teacher_path))

        # Creating a list of models to be input into an ensemble
        models = []

        # Each path needs to be separated by a ":"
        for path in self.args.teacher_path.split(":"):

            # Load ensemble member 
            model = utils.loaders.load_model(self.args, attribute = "teacher_arch").to(device)
            logger.info("Teacher member number of parameters: {:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1.0e6))

            # Checkpointer for loading
            checkpointer = utils.Checkpointer(self.args, path = path, save_last_n = -1)
            model.load_state_dict(checkpointer.load(device)['state_dict'])

            # Detach model parameters
            for param in model.parameters():
                param.detach_()

            # Append to list
            models.append(model)

        # Create ensemble
        return Ensemble(models)

    def get_correlation_metrics(self, input_logits, target_logits, all_target_logits):

        input_scalars = get_dirichlet_mutual_information(input_logits)
        input_scalars = input_scalars.detach().clone().cpu()
        target_logits = target_logits.detach().clone()
        all_target_logits = all_target_logits.detach().clone()

        # Get the target scalars
        target_scalars = get_mutual_information(target_logits, all_target_logits).cpu()

        # Compute correlations
        spear = scipy.stats.spearmanr(input_scalars, target_scalars)[0]
        pears = scipy.stats.pearsonr(input_scalars, target_scalars)[0]
        return spear, pears

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass and get the prediction info dictionary
        pred_l, _ = self.model(x_l)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_l, teacher_info = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_logits = pred_l, 
            target_logits = teacher_info['pred'],
            temperature = self.distillation_t,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(
            input_logits = pred_l, 
            target_logits = teacher_l, 
            all_target_logits = teacher_info['pred'],
        )

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
        teacher_l, teacher_info = self.teacher(x_l)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_l, teacher_l, self.distillation_t)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_logits = pred_l, 
            target_logits = teacher_info['pred'],
            temperature = self.distillation_t,
        )

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2
        loss += proxy * self.proxy_w

        # Compute correlation
        spear, pears = self.get_correlation_metrics(
            input_logits = pred_l, 
            target_logits = teacher_l, 
            all_target_logits = teacher_info['pred']
        )

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


def crossentropy_and_dirichlet_distillation(**kwargs):
    return DirichletDistillation(**kwargs)
