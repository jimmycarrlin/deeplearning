from typing import Optional

import pytest

import torch
from torch import nn

from src.detr.masked_tensor.masked_tensor import MaskedTensor
from src.detr.modules.encoder import TransformerEncoder, TransformerEncoderLayer
from src.detr.modules.decoder import TransformerDecoder, TransformerDecoderLayer


class Transformer(nn.Module):

    def __init__(self,
                 Nh: int,
                 d: int,
                 W_max: int,
                 H_max: int,
                 N_max: int = 100,
                 d_ffn: int = 2048,
                 dropout: float = 0.1,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dk: Optional[int] = None,
                 dv: Optional[int] = None):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(Nh, d, d_ffn, dropout, dk, dv)
        decoder_layer = TransformerDecoderLayer(Nh, d, d_ffn, dropout, dk, dv)

        self.encoder = TransformerEncoder(num_encoder_layers, encoder_layer, d, W_max, H_max)
        self.decoder = TransformerDecoder(num_decoder_layers, decoder_layer, d, W_max, H_max, N_max)

    def forward(self, ftr_maps: MaskedTensor, obj_queries: MaskedTensor):
        memory = self.encoder(ftr_maps)
        output = self.decoder(obj_queries, memory)
        return output

    def extra_repr(self):
        return f"{self._get_name()}({self.encoder}, {self.decoder})"


@pytest.fixture
def ftr_maps():
    yield MaskedTensor([
        torch.rand(7, 7, 256),
        torch.rand(6, 7, 256),
    ])


@pytest.fixture
def obj_queries():
    yield MaskedTensor(torch.rand(2, 100, 256))


@pytest.fixture
def transformer():
    yield Transformer(Nh=8, d=256, W_max=7, H_max=7)


def test_forward(transformer, ftr_maps, obj_queries):
    assert transformer(ftr_maps, obj_queries).size() == obj_queries.size()
