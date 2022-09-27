import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from loss.base import accuracy
from loss.distillation.distillation import Distillation, kl_divergence_loss

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
        self.use_training_data = args.distil_train_data
                                          
    def forward(self, info):
        # Get unlabelled batch
        x = info['x_ul'] if not self.use_training_data else info['x_l']
        
        # Perform model forward pass
        pred, _ = self.model(x)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_pred, _ = self.teacher(x)

        # Get the kl-loss averaged over batch
        kd = self.consistency_loss(pred, teacher_pred, self.distillation_t)

        # Compute total loss
        loss = kd * self.distillation_t ** 2

        # Compute accuracy for logging
        y = info['y_ul'] if not self.use_training_data else info['y_l']
        acc  = accuracy(pred.detach().clone(), y, top_k = (1, 5))

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'kd': kd.item(),
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}
        return loss, linfo

def distillation_unlabelled(**kwargs):
    return UnlabelledDistillation(**kwargs)