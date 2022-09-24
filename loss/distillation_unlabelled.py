import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from .base import accuracy
from .distillation import Distillation, kl_divergence_loss

__all__ = ['unlabelled_distillation']

class UnlabelledDistillation(Distillation):
    def __init__(self, args, model, optimiser, scheduler):
        super(Distillation, self).__init__(args, model, optimiser, scheduler)

        # Get loss specific arguments for knowledge distillation
        self.distillation_t  = args.temperature           # temperature used in loss

        # Build an ema teacher
        self.teacher = self.build_teacher()

        # Consistency loss
        self.consistency_loss = kl_divergence_loss

        # whether to use training data or unseen unlabelled data for distillation
        self.use_training_data = args.distil-train-data
                                          
    def forward(self, info):
        # Get unlabelled batch
        x_ul = info['x_ul'] if False else info['x']

        # Perform model forward pass
        pred_ul, _ = self.model(x_ul)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_ul, _ = self.teacher(x_ul)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred_ul, teacher_ul, self.distillation_t)

        # Compute total loss
        loss = kd * self.distillation_t ** 2

        # Compute accuracy
        acc = accuracy(pred_ul.detach().clone(), y_l, top_k = (1, 5))

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'ce': ce.item(),
            'kd': kd.item(),
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}
        return loss, linfo

def unlabelled_distillation(**kwargs):
    return UnlabelledDistillation(**kwargs)