from typing import Optional
import pytest

import torch
from torch import nn
from torch import Tensor
from torch.nn import Dropout
from torch.nn.modules.transformer import LayerNorm

from src.detr.modules.ffn import FFN
from src.detr.modules.attention.attention2d import MultiheadAttention2D
from src.detr.modules.positional_encoders.positional_encoders import _get_pos_encoder
from src.detr.util.model import _get_clones
from src.detr.masked_tensor.masked_tensor import MaskedTensor


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 Nh: int,
                 d: int,
                 d_ffn: int = 2048,
                 dropout: float = 0.1,
                 dk: Optional[int] = None,
                 dv: Optional[int] = None):

        super().__init__()

        self.attn = MultiheadAttention2D(Nh, d, dropout, dk, dv)

        self.ffn = FFN(d, d_ffn, dropout)

        self.norm1 = LayerNorm(d, eps=1e-5)
        self.norm2 = LayerNorm(d, eps=1e-5)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, q: Tensor, q_mask: Tensor, pos_logits: Tensor):
        shortcut = q

        output = self.attn(q, q, q_mask, q_mask, pos_logits)
        output = self.dropout1(output) + shortcut
        output = self.norm1(output)

        shortcut = output

        output = self.ffn(output)
        output = self.dropout2(output) + shortcut
        output = self.norm2(output)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers: int, layer: TransformerEncoderLayer, d: int, W_max: int, H_max: int):

        super().__init__()

        self.layers = _get_clones(layer, num_layers)
        self.pos_encoder = _get_pos_encoder("rl2d", d, W_max, H_max)

    def forward(self, q: MaskedTensor):
        q, q_mask = q.unbind()

        for layer in self.layers:
            pos_logits = self.pos_encoder(q)
            q = layer(q, q_mask, pos_logits)

        return MaskedTensor.bind(q.flatten(1, 2), q_mask.flatten(1, 2))


@pytest.fixture
def q():
    yield MaskedTensor([
        torch.rand(7, 7, 256),
        torch.rand(6, 7, 256),
    ])


@pytest.fixture
def layer():
    yield TransformerEncoderLayer(Nh=8, d=256)


@pytest.fixture
def encoder(layer):
    yield TransformerEncoder(num_layers=6, layer=layer, d=256, W_max=7, H_max=7)


def test_encoder_forward(encoder, q):
    m = encoder(q)
    assert m.size() == torch.Size([2, 49, 256])




