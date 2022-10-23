import torch.nn as nn

from .base import BaseLoss
from .base import accuracy

__all__ = ['sample_uncertainty_crossentropy']


def sample_crossentropy_and_entropy(info, target_labels):
    
    # Eross-entropy loss function with mean reduction
    ce = nn.CrossEntropyLoss()
    
    # Get the samples from info and average in log-domain (batch, classes)
    samples = info['samples'].mean(dim = 0)

    # Evaluate loss
    loss = ce(samples, target_labels)

    # Evaluate entropy of sampler and sum over all classes
    entropy = info['distribution'].entropy().sum(-1).mean()

    return loss, entropy


class MCUCE(BaseLoss):
    def __init__(self, args, model, optimiser, scheduler):
        super(MCUCE, self).__init__(args, model, optimiser, scheduler)

        # Eross-entropy loss function with mean reduction
        self.ce = nn.CrossEntropyLoss()

        # Get loss specific arguments
        self.mu = args.uce_mu

    def forward_mcuce(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass
        pred_l, info = self.model(x_l)

        # Get loss and entropy
        uce, ent = sample_crossentropy_and_entropy(info, y_l)

        # Get final loss
        loss = uce - self.mu * ent

        # Compute accuracy
        acc = accuracy(pred_l.detach().clone(), y_l, top_k = (1, 5))

        # Record metrics
        info = {'metrics': {
            'loss': loss.item(),
            'acc1': acc[0].item(),
            'acc5': acc[1].item(),
            'uce': uce.item(),
            'ent': ent.item(),
            'ce': self.ce(pred_l, y_l).item(),
        }}

        return loss, info

    def forward(self, info):
        return self.forward_mcuce(info)


def sample_uncertainty_crossentropy(**kwargs):
    return MCUCE(**kwargs)