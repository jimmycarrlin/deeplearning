from torch import nn
from torch.nn import Linear, Dropout


class FFN(nn.Module):

    def __init__(self, d: int, d_ffn: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.linear1 = Linear(d, d_ffn)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_ffn, d)

    def forward(self, x):
        output = self.linear1(x)
        output = self.dropout(output)
        output = self.linear2(output)

        return output