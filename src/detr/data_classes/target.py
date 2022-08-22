from typing import Any, Tuple

import torch
from torch import Tensor
from torch import isnan, equal, int64

import pytest
from itertools import islice
from dataclasses import dataclass

from src.detr.modules.criterion.matcher import HungarianMatcher


class ObjectDetectionTarget:
    """Shape:
        - labels: (sum{T(n)}),
        - bboxes: (sum{T(n)}, 4),
        - sizes: (N)."""

    __slots__ = ["labels", "bboxes", "sizes", "matched_labels", "matched_bboxes"]

    def __init__(self, labels: Tensor, bboxes: Tensor, sizes: Tuple[int, ...]):
        self.labels = labels
        self.bboxes = bboxes
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, item):
        return next(islice(iter(self), item, None))

    def __iter__(self):
        return zip(self.iter_labels(), self.iter_bboxes())

    def iter_labels(self):
        return self.labels.split(self.sizes)

    def iter_bboxes(self):
        return self.bboxes.split(self.sizes)

    @property
    def device(self):
        assert self.labels.device == self.bboxes.device
        return self.labels.device

    def to(self, device):
        return ObjectDetectionTarget(*[getattr(self, attr).to(device) for attr in ("labels", "bboxes")], self.sizes)

    def detach(self):
        return ObjectDetectionTarget(*[getattr(self, attr).detach() for attr in ("labels", "bboxes")], self.sizes)

    def cpu(self):
        return ObjectDetectionTarget(*[getattr(self, attr).cpu() for attr in ("labels", "bboxes")], self.sizes)


@pytest.fixture
def target():
    from src.detr.data_classes.target import ObjectDetectionTarget
    yield ObjectDetectionTarget(
        labels=torch.tensor([0, 0, 1, 1], dtype=int64),
        bboxes=torch.tensor(
            [[0.0,  0.0,  44.8,  44.8],
             [44.8, 0.0,  134.3, 89.6],
             [44.8, 44.8, 134.4, 134.4],
             [44.8, 89.6, 134.4, 179.2]]),
        sizes=(1, 3),
    )


def test_iter(target):
    it = iter(target)

    assert equal(next(it)[0], target.labels[:1])
    assert equal(next(it)[1], target.bboxes[1:, :])
    with pytest.raises(StopIteration):
        next(it)


def test_getitem(target):
    assert equal(target[0][0], target.labels[:1])
    assert equal(target[1][1], target.bboxes[1:, :])
    with pytest.raises(StopIteration):
        target[2]


def test_cuda(target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = target.to(device)
    assert target.labels.device.type == device.type

    target = target.detach().cpu()
    assert ~isnan(target.labels).all().item()


def test_mps(target):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    target = target.to(device)
    assert target.labels.device.type == device.type

    target = target.detach().cpu()
    assert ~isnan(target.labels).all().item()


def test_matched_labels(target):
    with pytest.raises(AttributeError):
        target.matched_labels


def test_matched_bboxes(target):
    with pytest.raises(AttributeError):
        target.matched_bboxes
