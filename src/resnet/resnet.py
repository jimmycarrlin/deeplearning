from functools import partial

from tqdm.notebook import tqdm, trange

import matplotlib

import torch
from torch.nn import BatchNorm2d, AdaptiveAvgPool2d, Linear
from torch import nn
from torch.nn.functional import softmax

from torchmetrics import Accuracy
from torchmetrics.classification.stat_scores import StatScores

from src.plotting import plot_random_preds
from src.context_managers import eval_mode, train_mode
from src.callback import DefaultCallback


# Aliases
Conv7x7 = partial(nn.Conv2d, kernel_size=7, padding=3, bias=False)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1, bias=False)
Conv1x1 = partial(nn.Conv2d, kernel_size=1, padding=0, bias=False)
MaxPool3x3 = partial(nn.MaxPool2d, kernel_size=3, padding=1)
ReLU = partial(nn.ReLU, inplace=True)

etqdm = partial(trange, unit="epoch", desc="Epoch loop")
btqdm = partial(tqdm, unit="batch", desc="Batch loop", leave=False)


class BasicBlock(nn.Module):
    """Basic block for ResNet."""
    expansion = 1

    def __init__(self, in_channels, hid_channels, stride=1):
        super().__init__()

        self.conv1 = Conv3x3(in_channels, hid_channels, stride=stride)
        self.bn1 = BatchNorm2d(hid_channels)
        self.conv2 = Conv3x3(hid_channels, hid_channels, stride=1)
        self.bn2 = BatchNorm2d(hid_channels)
        self.relu = ReLU()

        if (stride != 1) or (in_channels != self.expansion * hid_channels):
            self.downsample = nn.Sequential(
                Conv1x1(in_channels, self.expansion * hid_channels, stride=stride)
            )
        else:
            self.downsample = torch.nn.Sequential()  # Identity

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet."""
    expansion = 4

    def __init__(self, in_channels, hid_channels, stride=1):
        super().__init__()

        self.conv1 = Conv1x1(in_channels, hid_channels, stride=1)
        self.bn1 = BatchNorm2d(hid_channels)
        self.conv2 = Conv3x3(hid_channels, hid_channels, stride=stride)
        self.bn2 = BatchNorm2d(hid_channels)
        self.conv3 = Conv1x1(hid_channels, hid_channels * self.expansion, stride=1)
        self.bn3 = BatchNorm2d(hid_channels * self.expansion)
        self.relu = ReLU()

        if (stride != 1) or (in_channels != self.expansion * hid_channels):
            self.downsample = nn.Sequential(
                Conv1x1(in_channels, self.expansion * hid_channels, stride=stride)
            )
        else:
            self.downsample = nn.Sequential()  # Identity

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet configurable by block class (BasicBlock or Bottleneck) and number of blocks in each of 4 layers."""
    def __init__(self, block_cls, layers, num_classes):
        super().__init__()
        self.block_cls = block_cls
        self.layers = layers
        self.num_classes = num_classes
        self.in_channels = 64

        self.conv1 = Conv7x7(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool3x3(stride=2)
        self.layer1 = self._make_layer(block_cls, 64, layers[0], first_stride=1)
        self.layer2 = self._make_layer(block_cls, 128, layers[1], first_stride=2)
        self.layer3 = self._make_layer(block_cls, 256, layers[2], first_stride=2)
        self.layer4 = self._make_layer(block_cls, 512, layers[3], first_stride=2)
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(512 * block_cls.expansion, num_classes)

    def _make_layer(self, block_cls, hid_channels, num_blocks, first_stride):
        layer = nn.Sequential()

        layer.add_module(
            '0', block_cls(self.in_channels, hid_channels, first_stride)
        )
        self.in_channels = hid_channels * block_cls.expansion

        for i in range(1, num_blocks):
            layer.add_module(
                str(i), block_cls(self.in_channels, hid_channels, stride=1)
            )

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def extra_repr(self):
        num_layers = sum(self.layers) * (2 if self.block_cls == BasicBlock else 3) + 2
        return f"{self._get_name()}{num_layers}"


# Standard architectures
def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(BasicBlock, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(BasicBlock, [3, 8, 36, 3], num_classes)
