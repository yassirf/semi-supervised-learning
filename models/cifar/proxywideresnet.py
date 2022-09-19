import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wideresnet import WideResNet

__all__ = [
    'proxy_wideresnet162',
    'proxy_wideresnet282',
]


class TwoHeadWideResNet(WideResNet):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, survival=1, survival_mode='constant', **kwargs):
        super(TwoHeadWideResNet, self).__init__(depth, widen_factor, dropout_rate, num_classes, survival, survival_mode, **kwargs)

        # Define a secondary head
        self.linear2 = nn.Linear(self.n_stages[3], 1)

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
            'proxy': out2.squeeze(-1), 
        }
        return out, info


def proxy_wideresnet162(**kwargs):
    return TwoHeadWideResNet(
        depth = 16, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        **kwargs
    )


def proxy_wideresnet282(**kwargs):
    return TwoHeadWideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        **kwargs
    )
