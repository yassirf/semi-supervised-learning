import torch
import torch.nn as nn

from normalizingflow import InverseAutoregressiveFlow, RadialFlow
from wideresnet import wideresnet282, wideresnet2810, wideresnet404

__all__ = [
    'iafwideresnet282',
    'iafwideresnet2810',
    'iafwideresnet404',
    'radialwideresnet282',
    'radialwideresnet2810',
    'radialwideresnet404',
]


class PosteriorNetwork(nn.Module):
    def __init__(self, backbone, latent_dim, flow_length, num_classes, counts, **kwargs):
        super(PosteriorNetwork, self).__init__()

        # Number of classes
        self.num_classes = num_classes

        # Number of examples per class
        self.log_counts = counts.view(1, -1).log()

        # Backbone model mapping inputs to a low-dimensional space
        self.backbone = backbone(num_classes = latent_dim, **kwargs)
        self.backbonebn = nn.BatchNorm1d(num_features = latent_dim)

        # Normalizing flow for each class
        self.density = None

        # Softplus for numerical stability
        self.softplus = nn.Softplus(beta = 1, threshold = 20)

    def forward(self, x):
        
        # Get latent representation
        z = self.backbone(x)
        z = self.backbonebn(z)

        # Evaluate density for each class (batch, num_classes)
        log_probs = torch.stack([self.density[c].log_prob(z) for c in range(self.num_classes)], dim = 1)

        # Evaluate the dirichlet parameters from density (batch, num_classes)
        log_alphas = self.softplus(log_probs + self.log_counts)

        # Create dictionary with additional information
        info = {'pred': log_alphas, 
                'log_probs_nf': log_probs}

        return log_alphas, info


class IAFPosteriorNetwork(PosteriorNetwork):
    def __init__(self, backbone, latent_dim, flow_length, num_classes, counts, **kwargs):
        super().__init__(backbone, latent_dim, flow_length, num_classes, counts, **kwargs)

        # Normalizing flow for each class
        self.density = nn.ModuleList([
            InverseAutoregressiveFlow(latent_dim = latent_dim, flow_length = flow_length, **kwargs) 
            for _ in range(num_classes)
        ])


class RadialPosteriorNetwork(PosteriorNetwork):
    def __init__(self, backbone, latent_dim, flow_length, num_classes, counts, **kwargs):
        super().__init__(backbone, latent_dim, flow_length, num_classes, counts, **kwargs)

        # Normalizing flow for each class
        self.density = nn.ModuleList([
            RadialFlow(latent_dim = latent_dim, flow_length = flow_length, **kwargs) 
            for _ in range(num_classes)
        ])


def iafwideresnet282(latent_dim, flow_length, num_classes, counts, **kwargs):
    return IAFPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        counts = counts,
        **kwargs
    )


def iafwideresnet2810(latent_dim, flow_length, num_classes, counts, **kwargs):
    return IAFPosteriorNetwork(
        backbone = wideresnet2810,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        counts = counts,
        **kwargs
    )


def iafwideresnet404(latent_dim, flow_length, num_classes, counts, **kwargs):
    return IAFPosteriorNetwork(
        backbone = wideresnet404,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        counts = counts,
        **kwargs
    )


def radialwideresnet282(latent_dim, flow_length, num_classes, counts, **kwargs):
    return RadialPosteriorNetwork(
        backbone = wideresnet282,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        counts = counts,
        **kwargs
    )


def radialwideresnet2810(latent_dim, flow_length, num_classes, counts, **kwargs):
    return RadialPosteriorNetwork(
        backbone = wideresnet2810,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        counts = counts,
        **kwargs
    )


def radialwideresnet404(latent_dim, flow_length, num_classes, counts, **kwargs):
    return RadialPosteriorNetwork(
        backbone = wideresnet404,
        latent_dim = latent_dim,
        flow_length = flow_length,
        num_classes = num_classes, 
        counts = counts,
        **kwargs
    )



if __name__ == '__main__':

    # Logger for script and useful timing information
    import logging

    # Load logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Load some model
    logger.info("Loading model wideresnet282")
    model = wideresnet282(num_classes = 100)
    logger.info("Number of model parameters: {}\n".format(sum(p.numel() for p in model.parameters())))

    # Load some model
    logger.info("Loading model iaf-wideresnet282")
    model = iafwideresnet282(latent_dim = 16, flow_length = 2, num_classes = 100, counts = torch.zeros(100) + 500)
    logger.info("Number of model parameters: {}\n".format(sum(p.numel() for p in model.parameters())))

    # Load some model
    logger.info("Loading model radial-wideresnet282")
    model = radialwideresnet282(latent_dim = 16, flow_length = 8, num_classes = 100, counts = torch.zeros(100) + 500)
    logger.info("Number of model parameters: {}\n".format(sum(p.numel() for p in model.parameters())))
