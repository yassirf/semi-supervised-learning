import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wideresnet import WideResNet

__all__ = [
    'exp_gaussian_wideresnet282',
    'soft_gaussian_wideresnet282',
    'exp_laplace_wideresnet282',
    'soft_laplace_wideresnet282',
]


class TwoHeadWideResNet(WideResNet):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, survival=1, survival_mode='constant', **kwargs):
        super(TwoHeadWideResNet, self).__init__(depth, widen_factor, dropout_rate, num_classes, survival, survival_mode, **kwargs)

        # Define a secondary head
        self.linear2 = nn.Linear(self.n_stages[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        
        # Pass through two separate linear layers
        out = self.linear(x)
        out2 = self.linear2(x)

        info = {
            'pred': out, 
            'lin1': out, 
            'lin2': out2,
        }
        return (out, out2), info


class GaussianWideResNet(TwoHeadWideResNet):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, survival=1, survival_mode='constant', scale_fn = 'exp', num_samples=1, **kwargs):
        super(GaussianWideResNet, self).__init__(depth, widen_factor, dropout_rate, num_classes, survival, survival_mode, **kwargs)

        # How to parametrise the scale
        self.scale_fn = torch.exp if scale_fn == 'exp' else nn.Softplus(beta = 1, threshold = 20)

        # Number of samples from logit distribution at inference time
        self.num_samples = num_samples

    def forward(self, x):

        # Get the outputs from system
        (out, out2), _ = super(GaussianWideResNet, self).forward(x)

        # Get the scale from one of the heads
        out2 = self.scale_fn(out2)

        # Gaussian Sampler
        sampler = torch.distributions.normal.Normal(out, out2)

        # Sample from gaussian with reparametrisation trick (num, batch, class)
        samples = sampler.rsample(sample_shape = (self.num_samples, ))
        samples = torch.log_softmax(out, dim = -1)

        # Apply averaging in probability space (batch, class)
        pred = torch.logsumexp(samples, dim = 0) - math.log(self.num_samples)

        info = {
            'pred': pred,
            'samples': samples,
            'distribution': sampler,
        }

        return pred, info


class LaplaceWideResNet(TwoHeadWideResNet):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, survival=1, survival_mode='constant', scale_fn = 'exp', num_samples=1, **kwargs):
        super(LaplaceWideResNet, self).__init__(depth, widen_factor, dropout_rate, num_classes, survival, survival_mode, **kwargs)

        # How to parametrise the scale
        self.scale_fn = torch.exp if scale_fn == 'exp' else nn.Softplus(beta = 1, threshold = 20)

        # Number of samples from logit distribution at inference time
        self.num_samples = num_samples

    def forward(self, x):

        # Get the outputs from system
        (out, out2), _ = super(LaplaceWideResNet, self).forward(x)

        # Get the scale from one of the heads
        out2 = self.scale_fn(out2)

        # Gaussian Sampler
        sampler = torch.distributions.laplace.Laplace(out, out2)

        # Sample from gaussian with reparametrisation trick (num, batch, class)
        samples = sampler.rsample(sample_shape = (self.num_samples, ))
        samples = torch.log_softmax(out, dim = -1)

        # Apply averaging in probability space (batch, class)
        pred = torch.logsumexp(samples, dim = 0) - math.log(self.num_samples)

        info = {
            'pred': pred,
            'samples': samples,
            'distribution': sampler,
        }

        return pred, info


def exp_gaussian_wideresnet282(**kwargs):
    return GaussianWideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        scale_fn='exp',
        **kwargs
    )


def soft_gaussian_wideresnet282(**kwargs):
    return GaussianWideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        scale_fn='soft',
        **kwargs
    )


def exp_laplace_wideresnet282(**kwargs):
    return LaplaceWideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        scale_fn='exp',
        **kwargs
    )


def soft_laplace_wideresnet282(**kwargs):
    return LaplaceWideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        scale_fn='soft',
        **kwargs
    )

