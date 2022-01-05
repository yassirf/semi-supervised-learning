import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np


__all__ = [
    'wideresnet282'
]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        # Single conv layer
        out = self.conv1(F.relu(self.bn1(x)))
        # Dropout
        out = self.dropout(out)
        # Single conv layer
        out = self.conv2(F.relu(self.bn2(out)))
        # Residual 
        out += self.shortcut(x)
        return out

        
class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, **kwargs):
        super(WideResNet, self).__init__()

        # Ensure the correct wide-resnet size
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6

        # Number of planes in each block
        n_stages = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self.in_planes = 16

        self.conv1 = conv3x3(in_planes = 3, out_planes = n_stages[0], stride = 1)
        self.layer1 = self._make_layer(n, BasicBlock, n_stages[1], dropout_rate, stride=1)
        self.layer2 = self._make_layer(n, BasicBlock, n_stages[2], dropout_rate, stride=2)
        self.layer3 = self._make_layer(n, BasicBlock, n_stages[3], dropout_rate, stride=2)

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

    def _make_layer(self, num_blocks, block, planes, dropout_rate, stride):
        layers = []
        for _ in range(num_blocks):
            # The first block will have the self.in_planes and stride
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            # Afterwards all remaining blocks will have in_planes = outplanes = planes
            self.in_planes = planes
            # And will also have a stride of 1
            stride = 1
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


def wideresnet2810(**kwargs):
    return WideResNet(
        depth = 28, 
        widen_factor = 10, 
        dropout_rate = 0.3, 
        **kwargs
    )


def wideresnet404(**kwargs):
    return WideResNet(
        depth = 40, 
        widen_factor = 4, 
        dropout_rate = 0.3, 
        **kwargs
    )