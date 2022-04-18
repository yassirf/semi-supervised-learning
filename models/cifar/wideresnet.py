import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np


__all__ = [
    'wideresnet282',
    'wideresnet2810',
    'wideresnet404',
    'linearwideresnet282',
    'linearwideresnet2810',
    'linearwideresnet404',
]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, survival=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                self.bn1,
                self.relu1,
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        # For stochastic depth
        self.survival = survival

    def forward(self, x):
        # For stochastic depth
        if self.training and torch.rand(1).item() > self.survival:
            return self.shortcut(x)
        # Single conv layer
        out = self.conv1(self.relu1(self.bn1(x)))
        # Single conv layer with dropout
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        # Residual and weighting of signal
        out = self.shortcut(x) + (out if self.training else self.survival * out)
        return out

        
class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, survival=1.0, survival_mode='constant', **kwargs):
        super(WideResNet, self).__init__()

        # Ensure the correct wide-resnet size
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6

        # For stochastic depth mode
        self.survival = 1.0 if survival_mode == 'linear' else survival
        self.survival_decrease = (1.0 - survival)/(3*n) if survival_mode == 'linear' else 0.0

        # Number of planes in each block
        n_stages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.in_planes = 16

        self.conv1 = conv3x3(in_planes = 3, out_planes = n_stages[0], stride = 1)        
        self.layer1 = self._make_layer(n, BasicBlock, n_stages[1], dropout_rate, stride=1, survival=self.survival)
        self.layer2 = self._make_layer(n, BasicBlock, n_stages[2], dropout_rate, stride=2, survival=self.survival)
        self.layer3 = self._make_layer(n, BasicBlock, n_stages[3], dropout_rate, stride=2, survival=self.survival)

        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], num_classes)
        self.__initialise()

    def __initialise(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, num_blocks, block, planes, dropout_rate, stride, survival=1.0):
        layers = []
        for _ in range(num_blocks):
            # The first block will have the self.in_planes and stride
            layers.append(block(self.in_planes, planes, dropout_rate, stride, survival = survival))
            # Afterwards all remaining blocks will have in_planes = outplanes = planes
            self.in_planes = planes
            # And will also have a stride of 1
            stride = 1
            # Update survival rate for next block
            survival -= self.survival_decrease
        # For the next net-block
        self.survival = survival
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        info = {'pred': out}
        return out, info


def wideresnet282(**kwargs):
    return WideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        **kwargs
    )


def linearwideresnet282(**kwargs):
    return WideResNet(
        depth = 28, 
        widen_factor = 2, 
        dropout_rate = 0.3, 
        survival = 0.50, 
        survival_mode = 'linear',
        **kwargs
    )


def wideresnet2810(**kwargs):
    return WideResNet(
        depth = 28, 
        widen_factor = 10, 
        dropout_rate = 0.3, 
        **kwargs
    )


def linearwideresnet2810(**kwargs):
    return WideResNet(
        depth = 28, 
        widen_factor = 10, 
        dropout_rate = 0.3, 
        survival = 0.50, 
        survival_mode = 'linear',
        **kwargs
    )


def wideresnet404(**kwargs):
    return WideResNet(
        depth = 40, 
        widen_factor = 4, 
        dropout_rate = 0.3, 
        **kwargs
    )


def linearwideresnet404(**kwargs):
    return WideResNet(
        depth = 40, 
        widen_factor = 4, 
        dropout_rate = 0.3, 
        survival = 0.50,
        survival_mode = 'linear',
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

    # Print model hiden size and number of parameters
    for name, param in model.named_parameters():
        if name == 'linear.weight': logger.info("Final hidden layer size (out, in): {}".format(param.size()))
    logger.info("Number of model parameters: {}\n".format(sum(p.numel() for p in model.parameters())))
