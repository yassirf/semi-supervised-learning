import torch
import torch.nn as nn
import torch.distributions as tdist

from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive

__all__ = [
    'InverseAutoregressiveFlow',
    'RadialFlow',
]


class ListFlow(nn.Module):
    def __init__(self, latent_dim, flow_length, **kwargs):
        super(ListFlow, self).__init__()

        # For gaussian prior on flow
        self.mean = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(latent_dim), requires_grad=False)

        # Normalizing flow invertible transformation
        self.transforms = None

    def forward(self, z):

        # Forward pass of flow gives both sample and likelihood
        log_jacob = 0

        for transform in self.transforms:
            # Transform z
            z_next = transform(z)

            # Add the log determinant of jacobian
            log_jacob += transform.log_abs_det_jacobian(z, z_next)

            # Update
            z = z_next

        return z, log_jacob

    def log_prob(self, x):

        # Get the latent and 
        z, log_jacob = self.forward(x)

        # Evaluate likelihood
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)

        # Get log probability of sample
        log_prob_x = log_prob_z + log_jacob
        return log_prob_x


class InverseAutoregressiveFlow(ListFlow):

    def __init__(self, latent_dim, flow_length, **kwargs):
        super(InverseAutoregressiveFlow, self).__init__(latent_dim, flow_length, **kwargs)

        # Normalizing flow invertible transformation
        self.transforms = nn.Sequential(*(
            affine_autoregressive(latent_dim, hidden_dims=[128, 128]) for _ in range(flow_length)
        ))


class RadialFlow(ListFlow):
    
    def __init__(self, latent_dim, flow_length, **kwargs):
        super(RadialFlow, self).__init__(latent_dim, flow_length, **kwargs)

        # Normalizing flow invertible transformation
        self.transforms = nn.Sequential(*(
            Radial(latent_dim) for _ in range(flow_length)
        ))