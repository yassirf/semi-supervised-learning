import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

from .cross_entropy import CrossEntropy

__all__ = ['vat']


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    """
    Disables batchnorm tracking stats during adverserial optimisation
    """
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _normalise(vec):
    """
    Normalise the vector with respect to the euclidian norm
    """
    vec_size = (vec.size(0), -1, *(1 for _ in range(vec.dim() - 2)))
    vec /= LA.norm(vec.view(*vec_size), dim = 1, keepdim = True) + 1e-8
    return vec


def _entropy(pred):
    return -(pred * torch.exp(pred)).sum(-1).mean()


class VAT(CrossEntropy):
    def __init__(self, args, model, optimiser, scheduler):
        super(VAT, self).__init__(args, model, optimiser, scheduler)

        # Get loss specific arguments
        self.alpha = args.vat_alpha
        self.ent = args.vat_ent
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.ip = args.vat_ip

    def forward_vat(self, info):

        # Get unlabelled image
        x = info['x_ul']

        # Get target probabilities
        pred, _ = self.model(x)
        pred = F.log_softmax(pred, dim = -1)

        # Compute the entropy
        ent = _entropy(pred)

        # Get predicted probabilities
        pred = torch.exp(pred.detach().clone())

        with torch.no_grad():
            # Get target probabilities
            pred, _ = self.model(x)
            pred = F.softmax(pred, dim = -1)

        # Prepare random unit tensor for adverserial attack
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _normalise(d)

        # Deactivating batchnorm tracking
        with _disable_tracking_bn_stats(self.model):

            # Calculate adversarial direction
            for _ in range(self.ip):

                # This is our adverserial attack which needs gradients
                d.requires_grad_()

                # Compute prediction in the adverserial direction
                adv_pred, _ = self.model(x + self.xi * d)

                # We need the log-probabilities for the KL-divergence
                adv_logp = F.log_softmax(adv_pred, dim = -1)

                # The virtual adverserial loss
                adv_distance = F.kl_div(adv_logp, pred, reduction='batchmean')

                # Compute gradients
                adv_distance.backward()

                # Our new adverserial pertubabtion is the normalised gradient
                d = _normalise(d.grad)

                # Zero model gradients
                self.model.zero_grad()

            # Calculate the Local Distributional Smoothness loss
            adv_pred, _ = self.model(x + self.eps * d)
            adv_logp = F.log_softmax(adv_pred, dim = -1)
            lds = F.kl_div(adv_logp, pred, reduction='batchmean')

        info = {'metrics': {
            'lds': lds.item(),
            'ent': ent.item(),
        }}

        return lds + self.ent * ent, info

    def forward(self, info):
        # Get the cross-entropy loss and metrics
        ce, ce_info = super(VAT, self).forward(info)

        # Get the virtual adverserial training loss
        lds, vat_info = self.forward_vat(info)

        # Compute total loss
        loss = ce + self.alpha * lds

        # Record metrics
        ce_info['metrics']['loss'] = loss.item()
        ce_info['metrics']['ce']   = ce.item()
        ce_info['metrics']['lds']  = vat_info['metrics']['lds']
        ce_info['metrics']['ent']  = vat_info['metrics']['ent']

        return loss, ce_info


def crossentropy_and_vat(**kwargs):
    return VAT(**kwargs)