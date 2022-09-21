import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from .base import accuracy
from .distillation import Distillation

__all__ = ['crossentropy_and_unlabelled_distillation']


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


class UnlabelledDistillation(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(Distillation, self).__init__(args, model, optimiser, scheduler)

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Get unlabelled batch
        x_ul = info['x_ul']

        # Perform model forward pass
        pred_l, _ = self.model(x_l)
        pred_ul, _ = self.model(x_ul)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_ul, _ = self.teacher(x_ul)

        # Compute ce-loss
        ce = self.ce(pred_l, y_l)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_ul, teacher_ul, self.distillation_t)

        # Compute total loss
        loss = (1 - self.distillation_w) * ce + self.distillation_w * kd * self.distillation_t ** 2

        # Compute accuracy
        acc = accuracy(pred_l.detach().clone(), y_l, top_k = (1, 5))

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'ce': ce.item(),
            'kd': kd.item(),
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}

        return loss, linfo


def crossentropy_and_unlabelled_distillation(**kwargs):
    return UnlabelledDistillation(**kwargs)