from typing import Tuple, List, Union

import torch
from torch import Tensor, uint8

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


def resize(mask: Tensor, size: Union[Tuple[int, int], List[int]], interpolation=InterpolationMode.NEAREST):
    return F.resize(mask.unsqueeze(0).float(), size, interpolation=interpolation).to(uint8).squeeze(0)


def hflip(mask: Tensor):
    return F.hflip(mask.unsqueeze(0)).squeeze(0)


def pad(mask: Tensor, padding: Tuple[int]):
    target['masks'] = F.pad(target['masks'], (0, padding[0], 0, padding[1]))