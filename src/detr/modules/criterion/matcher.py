import pytest

from typing import Tuple

import torch
from torch import nn, equal, isclose, as_tensor, int64, float16, Tensor

from scipy.optimize import linear_sum_assignment as lsa

from src.detr.util.box import box_giou


class HungarianMatcher(nn.Module):

    def __init__(self, w_lbls: float = 1.0, w_bbox: float = 0.5, w_giou: float = 0.5):
        super().__init__()

        self.w_lbls = w_lbls
        self.w_bbox = w_bbox
        self.w_giou = w_giou

    @torch.no_grad()
    def forward(self, output, target):
        """B = batch_size, Q = num_queries, C = num_classes, T(b) = num_target_bounding_boxes (variable)."""
        B, Q, C = output.logits.size()

        # Out: (B * Q, C or 4)
        out_probas = output.probas.flatten(0, 1)
        out_bboxes = output.unnormalized_bboxes.flatten(0, 1)

        # Out: (sum{T(b)}, _ or 4)
        tgt_labels = target.labels
        tgt_bboxes = target.bboxes

        # Out: (B * Q, sum{T(b)})
        cost_lbls = -out_probas[:, tgt_labels]
        cost_bbox = torch.cdist(out_bboxes, tgt_bboxes, p=1)
        cost_giou = -box_giou(out_bboxes, tgt_bboxes)

        # Out: (B, Q, sum{T(b)})
        cost_matrix = self.w_bbox*cost_bbox + self.w_lbls*cost_lbls + self.w_giou*cost_giou
        cost_matrix = cost_matrix.view(B, Q, -1).cpu()

        # Out: (B, T(b), T(b))
        indices = (lsa(m[b]) for b, m in enumerate(cost_matrix.split(target.sizes, dim=-1)))
        indices = ((as_tensor(I, dtype=int64), as_tensor(J, dtype=int64)) for I, J in indices)

        # Out: (sum{T(b)})
        qry_indices, obj_indices = zip(*indices)
        qry_indices = self._get_setter_indices(qry_indices)

        matched_out_logits = output.logits
        matched_out_bboxes = output.unnormalized_bboxes[qry_indices]

        matched_tgt_labels = torch.full((B, Q), fill_value=C-1, dtype=int64, device=target.device)
        matched_tgt_labels[qry_indices] = torch.cat([l[I] for l, I in zip(target.iter_labels(), obj_indices)])
        matched_tgt_bboxes = torch.cat([b[I] for b, I in zip(target.iter_bboxes(), obj_indices)], dim=0)

        output.matched_logits, output.matched_bboxes = matched_out_logits, matched_out_bboxes
        target.matched_labels, target.matched_bboxes = matched_tgt_labels, matched_tgt_bboxes

        return output, target

    @staticmethod
    def _get_setter_indices(indices: Tuple[Tensor]):
        batches = torch.cat([torch.full_like(I, fill_value=i) for i, I in enumerate(indices)])
        indices = torch.cat([I for I in indices])
        return batches, indices


@pytest.fixture
def matcher():
    return HungarianMatcher()


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
             [3, 356, 448]])
    )


def test_forward(matcher, output, target):
    with pytest.raises(AttributeError):
        output.matched_bboxes

    output, target = matcher(output, target)
    assert equal(output.matched_bboxes.to(dtype=float16),
                 torch.tensor(
                     [[0.0, 0.0, 44.8, 44.8],
                      [44.8, 71.2, 134.4, 142.4],
                      [44.8, 0.0, 134.4, 71.2],
                      [44.8, 35.6, 134.4, 106.8]],
                     dtype=float16
                 )
    )
