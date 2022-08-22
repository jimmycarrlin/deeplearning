import sys

import pytest

from typing import Optional

import torch
from torch import nn
from torch import equal, isnan, float32
from torch import Tensor
from torch.nn import Linear, Dropout
import torch.nn.functional as F

from src.detr.masked_tensor.masked_tensor import MaskedTensor
from src.detr.util.tensors import batched_cartesian_product


class MultiheadAttention2D(nn.Module):
    """Multi-head Attention (MHA). If q = k, then MHA = MHSA."""

    def __init__(self,
                 Nh: int,
                 d: int,
                 dropout: float = 0.1,
                 dk: Optional[int] = None,
                 dv: Optional[int] = None):

        super().__init__()

        dk, dv = dk or d, dv or d
        dkh, dvh = dk // Nh, dv // Nh

        self.Nh, self.dk, self.dv, self.dkh, self.dvh = Nh, dk, dv, dkh, dvh

        self.projq = Linear(d, dk)
        self.projk = Linear(d, dk)
        self.projv = Linear(d, dv)

        self.linear = Linear(dv, d)
        self.dropout = Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, q_mask: Tensor, k_mask: Tensor, pos_logits: Tensor):
        assert q.device == k.device == q_mask.device == k_mask.device == pos_logits.device, \
            f"{q.device}, {k.device}, {q_mask.device}, {k_mask.device}"
        B, H, W, d = q.shape or k.shape

        # Out: (B, H, W, dk or dv)
        q = self.projq(q) * self.dkh ** -0.5
        k = self.projk(k)
        v = self.projv(k)

        # Out: (B, Nh, H, W, dkh or dvh)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Out: (B, 1, H, W, H, W)
        attn_mask = self._attn_mask(q_mask, k_mask)

        # Out: (B, Nh or 1, H, W, H, W)
        con_logits = torch.einsum("bhijd,bhkld->bhijkl", q, k)
        pos_logits = pos_logits.unsqueeze(1)

        # Out: (B, Nh, H, W, H, W)
        logits = attn_mask + con_logits + pos_logits

        # Out: (B, Nh, H, W, H, W)
        weights = logits.view((B, -1, H * W, H * W)).softmax(-1)
        weights = torch.reshape(weights, (B, -1, H, W, H, W))

        # Out: (B, Nh, H, W, dvh)
        attn = torch.einsum("bhijkl,bhkld->bhijd", weights, v)

        # Out: (B, H, W, dv)
        attn = self._combine_heads(attn)

        # Out: (B, H, W, d)
        attn = self.linear(attn)
        attn = self.dropout(attn)

        return attn

    def _split_heads(self, x: Tensor):
        """After splitting, shape is [B, Nh, H, W, dkh or dvh]."""
        B, H, W, d = x.shape
        Nh = self.Nh

        # Out: (B, L, Nh, dkh or dvh)
        split = torch.reshape(x, (B, H, W, Nh, d // Nh))

        # Out: (B, Nh, H, W, dkh or dvh)
        split = torch.permute(split, (0, 3, 1, 2, 4))

        return split

    @staticmethod
    def _combine_heads(x: Tensor):
        """Combine heads (inverse of '_split_heads')."""
        B, Nh, H, W, dvh = x.shape

        # Out: (B, H, W, Nh, dvh)
        x = torch.permute(x, (0, 2, 3, 1, 4))

        # Out: (B, H, W, dv)
        x = torch.reshape(x, (B, H, W, Nh * dvh))

        return x

    @staticmethod
    def _attn_mask(q_mask: Tensor, k_mask: Tensor):
        B, H, W = q_mask.shape or k_mask.shape
        device = q_mask.device or k_mask.device

        prod = batched_cartesian_product(q_mask.flatten(1, 2), k_mask.flatten(1, 2))
        fill_mask = torch.logical_xor(*prod.unbind(-1)).reshape(B, H, W, H, W)
        attn_mask = torch.zeros((B, H, W, H, W), dtype=float32, device=device).masked_fill_(fill_mask, float("-inf"))

        return attn_mask.unsqueeze(1)


@pytest.fixture
def mha():
    yield MultiheadAttention2D(Nh=8, d=256, dropout=0.0)


@pytest.fixture
def q():
    yield MaskedTensor([
        torch.rand(256, 7, 7),
        torch.rand(256, 6, 7),
    ])


@pytest.fixture
def pos_encoder():
    from src.detr.modules.positional_encoders import RelativeLearned2DPE
    yield RelativeLearned2DPE(d=256, W_max=7, H_max=7)


def test__attn_mask(mha, q):
    q, q_mask = q.unbind()

    attn_mask = mha._attn_mask(q_mask, q_mask)

    attn_mask1 = torch.zeros((7, 7, 7, 7), dtype=torch.float)
    attn_mask2 = torch.zeros((7, 7, 7, 7), dtype=torch.float)
    attn_mask2[6, :, :, :] = float("-inf")
    attn_mask2[:, :, 6, :] = float("-inf")

    assert equal(attn_mask, torch.stack([attn_mask1, attn_mask2]).unsqueeze(1))


def test__combine_heads_and__split_heads(mha, q):
    q, _ = q.unbind()
    q = q.permute(0, 2, 3, 1)
    assert equal(mha._combine_heads(mha._split_heads(q)), q)


def test_forward(mha, q, pos_encoder):
    q, q_mask = q.unbind()
    q = q.permute(0, 2, 3, 1)       # Channels last
    pos_logits = pos_encoder(q)

    assert mha(q, q, q_mask, q_mask, pos_logits).size() == torch.Size([2, 7, 7, 256])
