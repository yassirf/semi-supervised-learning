import math
import torch
import torch.nn as nn

from .normalizingflow import InverseAutoregressiveFlow, RadialFlow
from .wideresnet import wideresnet282, wideresnet2810, wideresnet404

__all__ = [
    'default_iafwideresnet282',
    'default_radialwideresnet282',
    'exp_iafwideresnet282',
    'exp_radialwideresnet282',
    'halfexp_iafwideresnet282',
    'halfexp_radialwideresnet282',
]


class EvidenceScaler(nn.Module):
    def __init__(self, dim, ctype = 'default') -> None:
        super().__init__()

        # Mapping from ctype to log-constant
        mapping = {
            'exp': 1.00,
            'halfexp': 0.50,
            'radial': 0.50 * math.log(4 * math.pi),
            'default': 0.50 * math.log(4 * math.pi),
        }

        # This determines the log-scale of counts
        # By default we set the scale to 
        self.log_scale = mapping[ctype] * dim

    def forward(self, log_evidence: torch.Tensor) -> torch.Tensor:
        return self.log_scale + log_evidence


class NaturalPosteriorNetwork(nn.Module):
    def __init__(self, backbone, latent_dim, flow_length, num_classes, count_type = 'default', **kwargs):
        super(NaturalPosteriorNetwork, self).__init__()

        # Number of classes
        self.num_classes = num_classes

        # Number of pseudo-examples
        self.scaler = EvidenceScaler(latent_dim, count_type)

        # Backbone model mapping inputs to a low-dimensional space
        self.backbone = backbone(num_classes = latent_dim, **kwargs)

        # Linear layer mapping to parameters
        self.linear = nn.Linear(latent_dim, num_classes)

        # Single normalizing flow
        self.density = None

        # Softplus for prior of dirichlet and numerical stability
        self.softplus = nn.Softplus(beta = 1, threshold = 20)

    def forward(self, x):
        
        # Get batch information
        batch = x.size(0)

        # Get latent representation
        z, _ = self.backbone(x)
        
        # Get parameters and ensure normalisation is correct
        log_probs = self.linear(z)
        log_probs = torch.log_softmax(log_probs, dim = -1)

        # Get log-evidence count
        log_evidence = self.scaler(self.density.log_prob(z))

        # Evaluate the dirichlet parameters from density (batch, num_classes)
        log_alphas = self.softplus(log_probs + log_evidence.view(batch, 1))

        # Create dictionary with additional information
        info = {
            'pred': log_alphas, 
            'log_probs_nf': log_probs
        }

        return log_alphas, info


class IAFNaturalPosteriorNetwork(NaturalPosteriorNetwork):
    def __init__(self, backbone, latent_dim, flow_length, num_classes, count_type = 'exp', **kwargs):
        super().__init__(backbone, latent_dim, flow_length, num_classes, count_type, **kwargs)

        # Normalizing flow for each class
        self.density = InverseAutoregressiveFlow(latent_dim = latent_dim, flow_length = flow_length, **kwargs)


class RadialNaturalPosteriorNetwork(NaturalPosteriorNetwork):
    def __init__(self, backbone, latent_dim, flow_length, num_classes, count_type = 'exp', **kwargs):
        super().__init__(backbone, latent_dim, flow_length, num_classes, count_type, **kwargs)

        # Normalizing flow for each class
        self.density = RadialFlow(latent_dim = latent_dim, flow_length = flow_length, **kwargs) 



def default_iafwideresnet282(latent_dim, flow_length, num_classes, **kwargs):
    return IAFNaturalPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        count_type = 'default',
        **kwargs
    )


def default_radialwideresnet282(latent_dim, flow_length, num_classes, **kwargs):
    return RadialNaturalPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        count_type = 'default',
        **kwargs
    )


def exp_iafwideresnet282(latent_dim, flow_length, num_classes, **kwargs):
    return IAFNaturalPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        count_type = 'exp',
        **kwargs
    )


def exp_radialwideresnet282(latent_dim, flow_length, num_classes, **kwargs):
    return RadialNaturalPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        count_type = 'exp',
        **kwargs
    )


def halfexp_iafwideresnet282(latent_dim, flow_length, num_classes, **kwargs):
    return IAFNaturalPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        count_type = 'halfexp',
        **kwargs
    )


def halfexp_radialwideresnet282(latent_dim, flow_length, num_classes, **kwargs):
    return RadialNaturalPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        count_type = 'halfexp',
        **kwargs
    )