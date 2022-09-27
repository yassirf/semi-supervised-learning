import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from loss.base import accuracy
from loss.distillation.distillation import Distillation
from loss.distillation.distillation import kl_divergence_loss


__all__ = [
    'crossentropy_and_unlabelled_distillation'
]


class UnlabelledDistillation(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(UnlabelledDistillation, self).__init__(args, model, optimiser, scheduler)

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']
        
        # Get unlabelled image 
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

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_ul, teacher_ul)

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'ce': ce.item(),
            'kd': kd.item(),
            'spear': spear,
            'pears': pears,
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}

        return loss, linfo


def crossentropy_and_unlabelled_distillation(**kwargs):
    return UnlabelledDistillation(**kwargs)