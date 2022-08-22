import pytest

from typing import Optional

import torch
from torch import equal
from torch import nn
from torch import Tensor
from torch.nn import Sequential, Conv2d

import torch.nn.functional as F
import torchvision
from torchvision.ops import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter

from src.detr.masked_tensor.masked_tensor import MaskedTensor
from src.detr.util.cnn_math import o
from src.detr.util.model import cnn_out_channels
from torchvision.models import resnet50


class Backbone(nn.Module):

    def __init__(self, model_name: str, d: int, pretrained: bool = True, train: bool = True):
        super().__init__()

        self.model_name = model_name
        self.d = d

        try:
            backbone = getattr(torchvision.models, model_name)
        except AttributeError:
            raise NotImplementedError

        layers = dict((f"layer{i}", f"layer{i}") for i in range(1, 5))

        backbone = backbone(pretrained=pretrained, norm_layer=FrozenBatchNorm2d).train(train)
        backbone = IntermediateLayerGetter(backbone, layers)

        in_proj = Conv2d(cnn_out_channels(backbone), d, kernel_size=1)

        self.body = Sequential(*[*backbone.children(), in_proj])

    def forward(self, imgs: MaskedTensor):
        feature_maps = self._forward_tensor(imgs)
        feature_maps_mask = self._forward_mask(imgs)
        return MaskedTensor.bind(feature_maps, feature_maps_mask)

    @torch.no_grad()
    def out_shapes(self, in_shapes: Tensor):
        out_shapes = []

        for C, H, W in in_shapes:
            for name, mod in self.body.named_modules():
                if isinstance(mod, nn.Conv2d) and ("downsample" not in name):
                    C = mod.out_channels if mod.in_channels == C else 1
                    H = o(H, mod.padding[0], mod.kernel_size[0], mod.stride[0])
                    W = o(W, mod.padding[1], mod.kernel_size[1], mod.stride[1])

                if isinstance(mod, nn.MaxPool2d) and ("downsample" not in name):
                    H = o(H, mod.padding, mod.kernel_size, mod.stride)
                    W = o(W, mod.padding, mod.kernel_size, mod.stride)

            out_shapes.append([C, H, W])

        return torch.tensor(out_shapes, dtype=torch.int, device=in_shapes.device)

    def out_channels(self):
        return cnn_out_channels(self.body)

    def _forward_tensor(self, imgs: MaskedTensor):
        tensor, _ = imgs.unbind()
        return self.body(tensor).permute(0, 2, 3, 1)

    def _forward_mask(self, imgs: MaskedTensor, reduce: Optional[str] = "crop"):
        _, mask = imgs.unbind()
        src_shapes, pad_shapes = imgs.src_shapes, imgs.pad_shapes

        if reduce == "convolve":
            return self._convolve_mask(mask)
        if reduce == "crop":
            return self._crop_mask(src_shapes, pad_shapes)
        if reduce == "interpolate":
            return self._interpolate_mask(mask, pad_shapes)

        raise NotImplementedError

    @torch.no_grad()
    def _convolve_mask(self, mask):
        # TODO
        return torch.tensor(0.0)

    @torch.no_grad()
    def _crop_mask(self, src_shapes: Tensor, pad_shapes: Tensor):
        src_shapes_conv = self.out_shapes(src_shapes)
        pad_shapes_conv = self.out_shapes(pad_shapes)

        B, (_, Hp, Wp) = len(pad_shapes_conv), pad_shapes_conv[0]

        mask_conv = torch.zeros(B, Hp, Wp, dtype=torch.uint8, device=pad_shapes.device)

        for src_shape, mask in zip(src_shapes_conv, mask_conv):
            _, Hg, Wg = src_shape.tolist()
            mask[:Hg, :Wg] = 1

        return mask_conv

    @torch.no_grad()
    def _interpolate_mask(self, mask: Tensor, pad_shapes: Tensor):
        _, *size = self.out_shapes(pad_shapes)[0]
        mask_conv = F.interpolate(mask.unsqueeze(1).float(), size=tuple(size)).to(torch.uint8).squeeze(1)
        return mask_conv

    def extra_repr(self):
        return f"{self.model_name}"


@pytest.fixture
def imgs():
    yield MaskedTensor([
        torch.rand(3, 224, 96),
        torch.rand(3, 180, 84),
    ])


@pytest.fixture
def backbone():
    yield Backbone("resnet50", d=256)


def test__interpolate_mask(backbone, imgs):
    mask, pad_shapes = imgs.mask, imgs.pad_shapes
    assert backbone._interpolate_mask(mask, pad_shapes).size() == torch.Size((2, 7, 3))


def test__crop_mask(backbone, imgs):
    src_shapes, pad_shapes = imgs.src_shapes, imgs.pad_shapes
    assert backbone._crop_mask(src_shapes, pad_shapes).size() == torch.Size((2, 7, 3))


def test__convolve_mask(backbone, imgs):
    ...


def test__forward_mask(backbone, imgs):
    mask = imgs.mask
    src_shapes, pad_shapes = imgs.src_shapes, imgs.pad_shapes
    assert equal(backbone._forward_mask(imgs, reduce="interpolate"), backbone._interpolate_mask(mask, pad_shapes))
    assert equal(backbone._forward_mask(imgs, reduce="crop"), backbone._crop_mask(src_shapes, pad_shapes))
    assert equal(backbone._forward_mask(imgs, reduce="convolve"), backbone._convolve_mask(mask))


def test__forward_tensor(backbone, imgs):
    feature_maps, _ = backbone(imgs).unbind()
    assert equal(backbone._forward_tensor(imgs), feature_maps)


def test_forward(backbone, imgs):
    feature_maps, feature_maps_mask = backbone(imgs).unbind()
    print(feature_maps.size())
    assert feature_maps.size()[-3:-1] == feature_maps_mask.size()[-2:]
