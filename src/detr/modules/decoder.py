from typing import Optional
import pytest

import torch
from torch import nn
from torch import Tensor
from torch.nn import Dropout
from torch.nn.modules.transformer import LayerNorm

from src.detr.masked_tensor.masked_tensor import MaskedTensor
from src.detr.modules.ffn import FFN
from src.detr.modules.attention.attention1d import MultiheadAttention
from src.detr.modules.positional_encoders.positional_encoders import _get_pos_encoder
from src.detr.util.model import _get_clones


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 Nh: int,
                 d: int,
                 d_ffn: int = 2048,
                 dropout: float = 0.1,
                 dk: Optional[int] = None,
                 dv: Optional[int] = None):

        super().__init__()

        self.attn1 = MultiheadAttention(Nh, d, dropout, dk, dv)
        self.attn2 = MultiheadAttention(Nh, d, dropout, dk, dv)

        self.ffn = FFN(d, d_ffn, dropout)

        self.norm1 = LayerNorm(d, eps=1e-5)
        self.norm2 = LayerNorm(d, eps=1e-5)
        self.norm3 = LayerNorm(d, eps=1e-5)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, q: Tensor, m: Tensor, q_mask: Tensor, m_mask: Tensor, pos_logits1: Tensor, pos_logits2: Tensor):
        shortcut = q

        output = self.attn1(q, q, q_mask, q_mask, pos_logits1)
        output = self.dropout1(output) + shortcut
        output = self.norm1(output)

        shortcut = output

        output = self.attn2(output, m, q_mask, m_mask, pos_logits2)
        output = self.dropout2(output) + shortcut
        output = self.norm2(output)

        shortcut = output

        output = self.ffn(output)
        output = self.dropout3(output) + shortcut
        output = self.norm3(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, num_layers: int, layer: TransformerDecoderLayer, d: int, W_max: int, H_max: int, N_max: int):

        super().__init__()

        self.layers = _get_clones(layer, num_layers)

        self.pos_encoder1 = _get_pos_encoder("al1d", d, N_max)
        self.pos_encoder2 = _get_pos_encoder("al1d", d, max(N_max, W_max * H_max))

    def forward(self, q: MaskedTensor, m: MaskedTensor):
        q, q_mask = q.unbind()
        m, m_mask = m.unbind()

        for layer in self.layers:
            pos_logits1 = self.pos_encoder1(q, q)
            pos_logits2 = self.pos_encoder2(q, m)

            q = layer(q, m, q_mask, m_mask, pos_logits1, pos_logits2)

        return q


@pytest.fixture
def q():
    yield MaskedTensor(torch.rand(2, 100, 256))


@pytest.fixture
def m():
    yield MaskedTensor([
        torch.rand(7, 7, 256),
        torch.rand(6, 7, 256),
    ])


@pytest.fixture
def layer():
    yield TransformerDecoderLayer(Nh=8, d=256)


@pytest.fixture
def decoder(layer):
    yield TransformerDecoder(num_layers=6, layer=layer, d=256, W_max=7, H_max=7, N_max=100)


def test_decoder_forward(decoder, q, m):
    assert decoder(q, m.flatten(1, 2)).size() == q.size()