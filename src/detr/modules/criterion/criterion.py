import pytest

import torch
from torch import nn
from torch import isclose
from torch.nn import functional as F

from src.detr.util.box import box_giou_loss
from src.detr.data_classes.output import ObjectDetectionOutput
from src.detr.data_classes.target import ObjectDetectionTarget
from src.detr.modules.criterion.matcher import HungarianMatcher


class SetCriterion(nn.Module):

    def __init__(self, num_classes: int, w_label: float = 1.0, w_bbox_l1: float = 1.0, w_bbox_giou: float = 1.0,
                 coef_eos: float = 0.1):

        super().__init__()

        self.num_classes = num_classes

        self.w_label = w_label
        self.w_bbox_l1 = w_bbox_l1
        self.w_bbox_giou = w_bbox_giou

        self.cls_weights = torch.ones(num_classes + 1)
        self.cls_weights[-1] = coef_eos

        self.matcher = HungarianMatcher()

    def forward(self, output: ObjectDetectionOutput, target: ObjectDetectionTarget):
        output, target = self.matcher(output, target)

        out_logits, out_bboxes = output.matched_logits, output.matched_bboxes
        tgt_labels, tgt_bboxes = target.matched_labels, target.matched_bboxes

        loss_labels = F.cross_entropy(out_logits.transpose(1, 2), tgt_labels, self.cls_weights.to(output.device))
        loss_bboxes_l1 = F.l1_loss(out_bboxes, tgt_bboxes)
        loss_bboxes_giou = box_giou_loss(out_bboxes, tgt_bboxes)

        return self.w_label*loss_labels + self.w_bbox_l1*loss_bboxes_l1 + self.w_bbox_giou*loss_bboxes_giou


@pytest.fixture
def criterion():
    yield SetCriterion(num_classes=2)


@pytest.fixture
def target():
    from src.detr.data_classes.target import ObjectDetectionTarget
    yield ObjectDetectionTarget(
        labels=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        bboxes=torch.tensor(
            [[0.0,  0.0,  44.8,  44.8],
             [44.8, 0.0,  134.3, 89.6],
             [44.8, 44.8, 134.4, 134.4],
             [44.8, 89.6, 134.4, 179.2]]),
        sizes=(1, 3),
    )


@pytest.fixture
def output():
    from src.detr.data_classes.output import ObjectDetectionOutput
    yield ObjectDetectionOutput(
        logits=torch.tensor(
            [[[1e6, 0.0, 0.0],
              [0.0, 0.0, 1e6],
              [0.0, 0.0, 1e6],
              [0.0, 0.0, 1e6]],

             [[0.0, 1e6, 0.0],
              [1e6, 0.0, 0.0],
              [0.0, 1e6, 0.0],
              [0.0, 0.0, 1e6]]]),
        bboxes=torch.tensor(
            [[[0.1, 0.1, 0.2, 0.2],
              [0.1, 0.2, 0.2, 0.2],
              [0.1, 0.3, 0.2, 0.2],
              [0.1, 0.4, 0.2, 0.2]],

             [[0.2, 0.3, 0.2, 0.2],
              [0.2, 0.1, 0.2, 0.2],
              [0.2, 0.2, 0.2, 0.2],
              [0.5, 0.5, 0.2, 0.2]]]),
        src_shapes=torch.tensor(
            [[3, 224, 224],
             [3, 448, 448]])
    )


def test_forward(criterion, output, target):
    assert isclose(criterion(output, target), torch.tensor(0.0), atol=0.01).item()
