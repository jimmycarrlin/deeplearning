import torch.nn.functional as F

from torch import nn
from torch.nn import Linear, Dropout, ModuleList


class MLP(nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, *, num_layers: int):
        super().__init__()

        self.num_layers = num_layers

        h = [hid_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if (i < self.num_layers - 1) else layer(x)

        return x
