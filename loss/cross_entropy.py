import torch.nn as nn

from .base import BaseLoss
from .base import accuracy

__all__ = ['crossentropy']


class CrossEntropy(BaseLoss):
    def __init__(self, args, model, optimiser, scheduler):
        super(CrossEntropy, self).__init__(args, model, optimiser, scheduler)

        # Eross-entropy loss function with mean reduction
        self.ce = nn.CrossEntropyLoss()

    def forward(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass
        pred_l, _ = self.model(x_l)

        # Compute loss
        loss = self.ce(pred_l, y_l)

        # Compute accuracy
        acc = accuracy(pred_l.detach().clone(), y_l, top_k = (1, 5))

        # Record metrics
        linfo = {'metrics': {
            'loss': loss.item(),
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
        }}

        return loss, linfo


def crossentropy(**kwargs):
    return CrossEntropy(**kwargs)