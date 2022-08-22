import pytest

from typing import Optional

import torch
from torch import nn
from torch import equal
from torch import Tensor
from torch.nn import Linear, Dropout
import torch.nn.functional as F

from src.detr.masked_tensor.masked_tensor import MaskedTensor


class MultiheadAttention(nn.Module):
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
        assert q.device == q_mask.device, f"{q.device} == {q_mask.device}"
        # Out: (B, Q or K, dk or dv)
        q = self.projq(q) * self.dkh ** -0.5
        k = self.projk(k)
        v = self.projv(k)

        # Out: (B, Nh, Q or K, dkh or dvh)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Out: (B, 1, Q, Q)
        attn_mask = self._attn_mask(q_mask, k_mask)

        # Out: (B, Nh or 1, Q, K)
        con_logits = torch.einsum("bhid,bhjd->bhij", q, k)
        pos_logits = pos_logits.unsqueeze(1)

        # Out: (B, Nh, Q, K)
        assert attn_mask.device == con_logits.device == pos_logits.device, f"{attn_mask.device} == {con_logits.device} == {pos_logits.device}"
        logits = attn_mask + con_logits + pos_logits

        # Out: (B, Nh, Q, K)
        weights = logits.softmax(-1)

        # Out: (B, Nh, Q, dvh)
        attn = torch.einsum("bhij,bhjd->bhid", weights, v)

        # Out: (B, Q, dv)
        attn = self._combine_heads(attn)

        # Out: (B, Q, d)
        attn = self.linear(attn)
        attn = self.dropout(attn)

        return attn

    def _split_heads(self, x: Tensor):
        """After splitting, shape is [B, Nh, H, W, dkh or dvh]."""
        B, L, d = x.shape
        Nh = self.Nh

        # Out: (B, L, Nh, dkh or dvh)
        split = torch.reshape(x, (B, L, Nh, d // Nh))

        # Out: (B, Nh, L, dkh or dvh)
        split = torch.permute(split, (0, 2, 1, 3))

        return split

    @staticmethod
    def _combine_heads(x: Tensor):
        """Combine heads (inverse of '_split_heads')."""
        B, Nh, L, dvh = x.shape

        # Out: (B, L, Nh, dvh)
        x = torch.permute(x, (0, 2, 1, 3))

        # Out: (B, L, dv)
        x = torch.reshape(x, (B, L, Nh * dvh))

        return x

    @staticmethod
    def _attn_mask(q_mask: Tensor, k_mask: Tensor):
        # Out: (B, Q, Q)
        attn_mask = torch.einsum("bi,bj->bij", q_mask, k_mask)
        attn_mask = attn_mask == 0

        attn_mask = torch.zeros_like(attn_mask, dtype=torch.float32).masked_fill_(attn_mask, float("-inf"))

        # Out: (B, 1, Q, Q), to be broadcastable with (B, Nh, Q, Q)
        attn_mask = torch.unsqueeze(attn_mask, dim=1)

        return attn_mask


@pytest.fixture
def mha():
    yield MultiheadAttention(Nh=8, d=256, dropout=0.0)


@pytest.fixture
def q():
    yield torch.rand(2, 100, 256)


@pytest.fixture
def k():
    yield MaskedTensor([
        torch.rand(256, 7, 7),
        torch.rand(256, 6, 7),
    ])


@pytest.fixture
def pos_encoder():
    from src.detr.modules.positional_encoders import AbsoluteLearnedPE
    yield AbsoluteLearnedPE(d=256, L_max=100)


def test__attn_mask(mha, q, k):
    q, q_mask = q, torch.ones(2, 100, dtype=torch.uint8)

    k, k_mask = k.unbind()
    k, k_mask = k.flatten(2, 3).permute(0, 2, 1), k_mask.flatten(1, 2)

    attn_mask = mha._attn_mask(q_mask, k_mask)

    attn_mask1 = torch.zeros((100, 49), dtype=torch.float)
    attn_mask2 = torch.zeros((100, 49), dtype=torch.float)
    attn_mask2[:, -7:] = float("-inf")

    assert equal(attn_mask, torch.stack([attn_mask1, attn_mask2]).unsqueeze(1))


def test__combine_heads_and__split_heads(mha, q, k):
    k, _ = k.unbind()
    k = k.flatten(2, 3).permute(0, 2, 1)

    assert equal(mha._combine_heads(mha._split_heads(q)), q)
    assert equal(mha._combine_heads(mha._split_heads(k)), k)


def test_forward(mha, q, k, pos_encoder):
    q, q_mask = q, torch.ones(2, 100)

    k, k_mask = k.unbind()
    k, k_mask = k.flatten(2, 3).permute(0, 2, 1), k_mask.flatten(1, 2)

    pos_logits = pos_encoder(q, k)

    assert mha(q, k, q_mask, k_mask, pos_logits).size() == torch.Size([2, 100, 256])
