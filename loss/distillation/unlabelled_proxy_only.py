import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from loss.base import accuracy
from loss.distillation.distillation_and_proxy import DistillationProxy
from loss.distillation.distillation import kl_divergence_loss


__all__ = [
    'unlabelled_proxy_rank'
]

class ProxyOnly(DistillationProxy):
    def __init__(self, args, model, optimiser, scheduler):
        super(ProxyOnly, self).__init__(args, model, optimiser, scheduler)

    def forward(self, info):

        # Get labelled image and label
        x_ul = info['x_ul']

        # Perform model forward pass and get the prediction info dictionary
        pred_ul, pred_info = self.model(x_ul)

        with torch.no_grad():
            # The second input is for the teacher model
            teacher_ul, _ = self.teacher(x_ul)

        # Get the proxy-loss 
        proxy = self.proxy_loss(
            input_scalars = pred_info['proxy'], 
            target_logits = teacher_ul, 
            param = self.proxy_regularization_strength,
        )

        # Compute total loss
        loss = proxy

        # Compute correlation
        spear, pears = self.get_correlation_metrics(pred_info['proxy'], teacher_ul)

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'kd': 0,
            'proxy': proxy.item(),
            'spear': spear,
            'pears': pears,
        }}

        return loss, linfo


def unlabelled_proxy_rank(**kwargs):
    return ProxyOnly(**kwargs)