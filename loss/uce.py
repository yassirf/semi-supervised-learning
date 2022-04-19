import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.distributions.dirichlet import Dirichlet

from .cross_entropy import CrossEntropy

__all__ = ['uncertainty_crossentropy']


def expected_crossentropy_and_entropy(log_alphas, target_labels):
    
    # Get batch size
    batch = target_labels.size(0)

    # Get parameters in real domain
    alphas = log_alphas.exp()

    # Get the alpha0 parameter for digamma input
    alpha0 = alphas.sum(-1)

    # Get the expected loss
    loss = torch.digamma(alpha0) - torch.digamma(alphas[torch.arange(batch), target_labels])
    
    # Get the entropy of dirichlet
    entropy = Dirichlet(alphas).entropy()

    return loss.mean(), entropy.mean()


class UCE(CrossEntropy):
    def __init__(self, args, model, optimiser, scheduler):
        super(UCE, self).__init__(args, model, optimiser, scheduler)

        # Eross-entropy loss function with mean reduction
        self.ce = nn.CrossEntropyLoss()

        # Get loss specific arguments
        self.mu = args.uce_mu

    def forward_uce(self, info):

        # Get labelled image and label
        x_l, y_l = info['x_l'], info['y_l']

        # Perform model forward pass
        pred_la, _ = self.model(x_l)

        # Get loss and entropy
        uce, ent = expected_crossentropy_and_entropy(pred_la, y_l)

        info = {'metrics': {
            'uce': uce.item(),
            'ent': ent.item(),
            'ce': self.ce(pred_la, y_l).item()
        }}

        return uce - self.mu * ent, info

    def forward(self, info):

        # Get the virtual adverserial training loss
        uce, uce_info = self.forward_uce(info)

        # Compute total loss
        loss = uce

        # Record metrics
        uce_info['metrics']['loss'] = loss.item()
        uce_info['metrics']['ce']   = uce_info['metrics']['ce']
        uce_info['metrics']['uce']  = uce_info['metrics']['uce']
        uce_info['metrics']['ent']  = uce_info['metrics']['ent']

        return loss, uce_info


def uncertainty_crossentropy(**kwargs):
    return UCE(**kwargs)