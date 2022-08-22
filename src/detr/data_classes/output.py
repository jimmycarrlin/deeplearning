import torch
from torch import Tensor
from torch import equal, isnan

import pytest
from dataclasses import dataclass

from src.detr.util.box import box_cxcywh_to_xyxy, box_rel_to_abs


class ObjectDetectionOutput:
    """Shape:
        - logits: (B, Q, C+1),
        - bboxes: (B, Q, 4},
        - src_shapes: (B, 2)"""

    __slots__ = ["logits", "bboxes", "src_shapes", "matched_logits", "matched_bboxes"]

    def __init__(self, logits: Tensor, bboxes: Tensor, src_shapes: Tensor):
        # TODO: default 'None' value for src_shapes.
        self.logits = logits
        self.bboxes = bboxes
        self.src_shapes = src_shapes

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, item):
        return self.logits[item], self.bboxes[item]

    def __iter__(self):
        return zip(self.iter_logits(), self.iter_bboxes())

    def iter_logits(self):
        return iter(self.logits)

    def iter_bboxes(self):
        return iter(self.bboxes)

    @property
    def probas(self):
        return self.logits.softmax(-1)

    @property
    def labels(self):
        _, labels = self.logits.max(-1)
        return labels

    @property
    def scores(self):
        scores, _ = self.probas[..., :-1].max(-1)
        return scores

    @property
    def normalized_bboxes(self):
        return self.bboxes

    @property
    def unnormalized_bboxes(self):
        return box_rel_to_abs(box_cxcywh_to_xyxy(self.bboxes), self.src_shapes)

    @property
    def device(self):
        assert self.logits.device == self.bboxes.device
        return self.logits.device

    def to(self, device):
        return ObjectDetectionOutput(*[getattr(self, attr).to(device) for attr in ("logits", "bboxes", "src_shapes")])

    def detach(self):
        return ObjectDetectionOutput(*[getattr(self, attr).detach() for attr in ("logits", "bboxes", "src_shapes")])

    def cpu(self):
        return ObjectDetectionOutput(*[getattr(self, attr).cpu() for attr in ("logits", "bboxes", "src_shapes")])


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


def test_iter(output):
    it = iter(output)

    assert equal(next(it)[0], output.logits[0, :])
    assert equal(next(it)[1], output.bboxes[1, :, :])
    with pytest.raises(StopIteration):
        next(it)


def test_getitem(output):
    assert equal(output[0][0], output.logits[0, :])
    assert equal(output[1][1], output.bboxes[1, :, :])
    with pytest.raises(IndexError):
        output[2]


def test_cuda(output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output = output.to(device)
    assert output.logits.device.type == device.type

    output = output.detach().cpu()
    assert ~isnan(output.logits).all().item()


def test_mps(output):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    output = output.to(device)
    assert output.logits.device.type == device.type

    output = output.detach().cpu()
    assert ~isnan(output.logits).all().item()


def test_matched_logits(output):
    with pytest.raises(AttributeError):
        output.matched_logits


def test_matched_bboxes(output):
    with pytest.raises(AttributeError):
        output.matched_bboxes
