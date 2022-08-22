from typing import List, Tuple, Dict, Optional, Union

import random
import PIL
import torch
import numpy as np

from torch import Tensor, as_tensor, uint8, float32

import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

from pycocotools.mask import frPyObjects as poly_to_rle, decode as rle_to_bin

import src.detr.util.box as B
import src.detr.util.mask as M


def nonzero_bbox(bbox: Tensor):
    return (bbox[..., :2] < bbox[..., 2:]).all(-1)


def nonzero_mask(mask: Tensor):
    return mask.flatten(-2, -1).any(-1)


def poly_to_mask(img: PIL.Image.Image, target: List[Dict]):
    w, h = img.size

    for obj in target:
        mask = rle_to_bin(poly_to_rle(obj["segmentation"], h, w))
        if mask.ndim < 3:
            mask = mask[..., None]
        obj["mask"] = mask.any(-1)

    return img, target


def to_tensor(img: Union[PIL.Image.Image, np.ndarray], target: List[Dict]):
    img = F.to_tensor(img)

    for obj in target:
        obj["bbox"] = as_tensor(obj["bbox"], dtype=float32)
        obj["mask"] = as_tensor(obj["mask"], dtype=uint8)

    return img, target


def normalize(img: Union[PIL.Image.Image, np.ndarray], target: List[Dict], mean: List[float], std: List[float]):
    return F.normalize(img, mean, std), target


def crop(img: Tensor, target: List[Dict], region: Tuple[int, ...]):  # TODO: if target is None
    cropped_img = F.crop(img, *region)
    cropped_target = []

    for obj in target:
        cropped_bbox = B.crop(obj["bbox"], *region)
        cropped_mask = F.crop(obj["mask"], *region)

        if nonzero_bbox(cropped_bbox) and nonzero_mask(cropped_mask):
            obj["bbox"], obj["mask"] = cropped_bbox, cropped_mask
            cropped_target.append(obj)

    if cropped_target:
        return cropped_img, cropped_target
    else:
        return img, target


def hflip(img: Tensor, target: List[Dict]):
    img = F.hflip(img)

    for obj in target:
        obj["bbox"] = B.hflip(obj["bbox"], img.shape[-1])
        obj["mask"] = M.hflip(obj["mask"])

    return img, target


def resize(img: Tensor, target: List[Dict], size: Union[int, Tuple[int, int]], max_size: int = None):
    size = _get_size(img.shape[-2:], size, max_size)
    height_ratio, width_ratio = (s / s_orig for s, s_orig in zip(size, img.shape[-2:]))

    img = F.resize(img, size)

    for obj in target:
        obj["bbox"] = B.resize(obj["bbox"], height_ratio, width_ratio)
        obj["mask"] = M.resize(obj["mask"], size)

    return img, target


def max_resize(img: Tensor, target: List[Dict], max_size: int):
    image_size = img.shape[-2:]
    if max(image_size) <= max_size:
        return img, target
    size = int(min(image_size) / max(image_size) * max_size)

    return resize(img, target, size)


def pad(img: Tensor, target: List[Dict], padding: Tuple[int, ...]):
    img = F.pad(img, padding)

    for obj in target:
        obj["mask"] = F.pad(obj["mask"], padding)

    return img, target


class PolyToMask:

    def __call__(self, img: Tensor, target: List[Dict]):
        return poly_to_mask(img, target)


class ToTensor:

    def __call__(self, img: Union[PIL.Image.Image, np.ndarray], target: List[Dict]):
        return to_tensor(img, target)


class Normalize:

    def __init__(self, mean: List[float] = None, std: List[float] = None):
        self.mean = mean if (mean is not None) else [0.485, 0.456, 0.406]
        self.std = std if (std is not None) else [0.229, 0.224, 0.225]

    def __call__(self, img: Union[Tensor, PIL.Image.Image], target: List[Dict]):
        return normalize(img, target, self.mean, self.std)


class CenterCrop:

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, img: Tensor, target: List[Dict]):
        img_height, img_width = img.shape
        crp_height, crp_width = self.size

        crp_top = int((img_height - crp_height) / 2)
        crp_left = int((img_width - crp_width) / 2)

        return crop(img, target, (crp_top, crp_left, crp_height, crp_width))


class Resize:

    def __init__(self, size: Union[int, Tuple[int, int]], max_size: Optional[int] = None):
        self.size = size
        self.max_size = max_size

    def __call__(self, img: Tensor, target: List[Dict]):
        return resize(img, target, self.size, self.max_size)


class MaxResize:

    def __init__(self, max_size: int):
        self.max_size = max_size

    def __call__(self, img: Tensor, target: List[Dict]):
        return max_resize(img, target, self.max_size)


class RandomCrop:

    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size

    def __call__(self, img: Tensor, target: List[Dict]):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop:

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Tensor, target: List[Dict]):
        h = random.randint(min(img.shape[-2], self.min_size), min(img.shape[-2], self.max_size))
        w = random.randint(min(img.shape[-1], self.min_size), min(img.shape[-1], self.max_size))
        region = T.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region)


class RandomHorizontalFlip:

    def __init__(self, p: Optional[float] = 0.5):
        self.p = p

    def __call__(self, img: Tensor, target: List[Dict]):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:

    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img: Tensor, target: List[Dict]):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad:

    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect:

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: Union[Tensor, PIL.Image.Image], target: List[Dict]):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def get_coco_transforms(mode: str):
    sizes = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if mode == "train":
        return Compose([
            PolyToMask(),
            ToTensor(),
            RandomHorizontalFlip(),
            RandomSelect(
                RandomResize(sizes, max_size=1333),
                Compose([
                    RandomResize([400, 500, 600]),
                    RandomSizeCrop(384, 600),
                    RandomResize(sizes, max_size=1333),
                ])
            ),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if mode == "val":
        return Compose([
            PolyToMask(),
            ToTensor(),
            RandomResize([800], max_size=1333),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    raise ValueError


def _get_size(
        image_size: Tuple[int, int],
        size: Union[int, Tuple[int, int]] = None,
        max_size: int = None):

    if isinstance(size, (tuple, list, torch.Size)):
        if (max_size is not None) and (max(size) < max_size):
            return size
        raise ValueError("'max(size)' must be less than max_size")

    if (max_size is not None) and (max_size < max(image_size) / min(image_size) * size):
        size = int(min(image_size) / max(image_size) * max_size)

    h, w = image_size
    if h == w:
        return size, size
    if h < w:
        return size, int(size * w / h)
    if h > w:
        return int(size * h / w), size
