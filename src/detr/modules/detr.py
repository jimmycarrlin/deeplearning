import pytest

import torch
from torch import nn
from torch.nn import Linear

from src.detr.modules.mlp import MLP
from src.detr.modules.transformer import Transformer
from src.detr.modules.backbone import Backbone
from src.detr.masked_tensor.masked_tensor import MaskedTensor
from src.detr.data_classes.output import ObjectDetectionOutput


class DETR(nn.Module):

    def __init__(self, backbone: nn.Module, transformer: nn.Module, d: int, num_classes: int):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer

        self.ffn1 = Linear(d, num_classes + 1)
        self.ffn2 = MLP(in_dim=d, hid_dim=d, out_dim=4, num_layers=3)

    def forward(self, imgs: MaskedTensor):
        ftr_maps = self.backbone(imgs)
        obj_queries = self.obj_queries()

        output = self.transformer(ftr_maps, obj_queries)
        logits = self.ffn1(output)
        bboxes = self.ffn2(output).sigmoid()  # TODO: Remove sigmoid.

        return ObjectDetectionOutput(logits, bboxes, imgs.src_shapes)

    def obj_queries(self):
        # TODO: Remove unsqueeze.
        return MaskedTensor(self.transformer.decoder.pos_encoder1.embed.unsqueeze(0))

    def parameters_to_optimize(self, lr_backbone: float = 1e-5):
        if lr_backbone > 0.0:
            return [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                 "lr": lr_backbone}
            ]

        if lr_backbone == 0.0:
            return [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]}
            ]

        raise ValueError(f"Invalid backbone learning rate: {lr_backbone}, it must be zero or positive.")

    @classmethod
    def default(cls, num_classes):
        backbone = Backbone("resnet50", d=256)
        transformer = Transformer(Nh=8, d=256, d_ffn=2048, W_max=50, H_max=50)

        return cls(backbone, transformer, d=256, num_classes=num_classes)

    def extra_repr(self):
        return f"{self._get_name()}_{self.backbone.extra_repr()}"


@pytest.fixture
def imgs():
    yield MaskedTensor([
        torch.rand(3, 224, 224),
        torch.rand(3, 448, 324),
    ], channels_last=False)


@pytest.fixture
def detr():
    yield DETR.default(num_classes=2)


def test_detr_forward(imgs, detr):
    output = detr(imgs)
    logits, bboxes = output.logits, output.bboxes

    assert logits.size() == torch.Size([2, 100, 2 + 1])
    assert bboxes.size() == torch.Size([2, 100, 4])
