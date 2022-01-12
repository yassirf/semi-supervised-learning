import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.beta as beta
from torch import linalg as LA

from .meanteacher import MeanTeacher
from .meanteacher import CrossEntropy

__all__ = ['crossentropy_and_ict']


def softmax_mse_loss(input_logits, target_logits):
    """
    Takes softmax on both sides and returns MSE loss
    Gradients only propagated through the input logits.
    """
    input_softmax = F.softmax(input_logits, dim = -1)
    target_softmax = F.softmax(target_logits, dim = -1)
    return F.mse_loss(input_softmax, target_softmax, reduction='mean')


class ICT(MeanTeacher):
    def __init__(self, args, model, optimiser, scheduler):
        # Note that ICT uses the same hyper-parameters as MeanTeacher
        super(ICT, self).__init__(args, model, optimiser, scheduler)

        # Get beta distribution parameter
        self.beta = beta.Beta(torch.tensor(args.mixup_alpha), torch.tensor(args.mixup_alpha))

    def forward_ict(self, info):
        
        # Get unlabelled images
        # For interpolation consistent training we split the batch into two subbatches
        x_ul = info['x_ul']
        x_ul = x_ul.view(-1, 2, *x_ul.size()[1:])

        # Sample mixup parameter to match shape of images (batch, height, width, channels)
        beta = self.beta.sample((x_ul.size(0), 1, 1, 1))

        # These two batches will be used in mixup
        xm_ul = beta * x_ul[:, 0] + (1 - beta) * x_ul[:, 1]

        # Also generate pseudo labels from teacher
        with torch.no_grad():

            # Reshape to match label sizes
            beta = beta.view(-1, 1)

            y_ul1, _ = self.teacher(x_ul[:, 0])
            y_ul2, _ = self.teacher(x_ul[:, 1])

            # The mixup target
            ym_ul = beta * y_ul1 + (1 - beta) * y_ul2

        # Generate student model prediction on mixup input
        pred_ul, _ = self.model(xm_ul)

        # Get the loss averaged over batch
        ict = self.consistency_loss(pred_ul, ym_ul)

        # Record metrics
        ictinfo = {'metrics': {'ict': ict.item()}}
        return ict, ictinfo

    def forward(self, info):
        # Get the cross-entropy loss and metrics
        ce, ce_info = super(MeanTeacher, self).forward(info)

        # Get the virtual adverserial training loss
        ict, _ = self.forward_ict(info)

        # Compute total loss
        loss = ce + self.w * ict

        # Record metrics
        ce_info['metrics']['loss'] = loss.item()
        ce_info['metrics']['ce']   = ce.item()
        ce_info['metrics']['ict']  = ict.item()

        return loss, ce_info


def crossentropy_and_ict(**kwargs):
    return ICT(**kwargs)