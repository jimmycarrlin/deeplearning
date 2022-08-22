from typing import Union, Optional, Iterable, List

import torch
from torch import Tensor, Size
from torch import uint8

from torch.utils.data import DataLoader


class MaskedTensor:

    def __init__(self, src: Union[Tensor, List[Tensor]], *, channels_last=True):

        if isinstance(src, Tensor):
            self.tensor = src
            self.mask = (torch.ones(src.shape[:-1]) if channels_last else torch.ones(src.shape[1:])).to(src.device)
            self._scr_shapes = self.pad_shapes
            return

        B = len(src)
        max_size = self._max_size(src)
        max_size_without_channels = max_size[:-1] if channels_last else max_size[1:]

        dtype = src[0].dtype
        device = src[0].device

        tensor = torch.zeros(B, *max_size, dtype=dtype, device=device)
        mask = torch.zeros(B, *max_size_without_channels, dtype=uint8, device=device)
        src_shapes = torch.empty(B, 3, dtype=torch.int)

        for src_tensor, src_shape, pad_tensor, pad_mask in zip(src, src_shapes, tensor, mask):

            if channels_last:
                H, W, C = src_tensor.size()
                pad_tensor[:H, :W, :C].copy_(src_tensor)
                pad_mask[:H, :W] = 1
                src_shape[:] = torch.tensor((H, W, C))

            else:
                C, H, W = src_tensor.size()
                pad_tensor[:C, :H, :W].copy_(src_tensor)
                pad_mask[:H, :W] = 1
                src_shape[:] = torch.tensor((C, H, W))

        self.tensor = tensor
        self.mask = mask
        self._src_shapes = src_shapes

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, item):
        return self.tensor[item], self.mask[item]

    def __iter__(self):
        return zip(iter(self.tensor), iter(self.mask))

    @classmethod
    def bind(cls, tensor: Tensor, mask: Tensor):
        assert tensor.device == mask.device

        instance = cls(src=torch.empty(0, device=tensor.device))

        instance.tensor = tensor
        instance.mask = mask
        instance._src_shapes = None

        return instance

    def unbind(self):
        return self.tensor, self.mask

    @property
    def src_shapes(self):
        if self._src_shapes is not None:
            return self._src_shapes

        B, C, *_ = self.tensor.shape

        shapes = torch.empty(B, 3, dtype=torch.int, device=self.tensor.device)

        for mask, shape in zip(self.mask, shapes):
            H = self._len_ones(mask[:, 0])
            W = self._len_ones(mask[0, :])
            shape[:] = torch.tensor((C, H, W))

        return shapes

    @property
    def pad_shapes(self):
        B, *shape = self.tensor.shape
        return torch.tensor(shape, dtype=torch.int, device=self.tensor.device).unsqueeze(0).repeat(B, 1)

    def size(self):
        return self.tensor.size()

    @property
    def shape(self):
        return self.size()

    def flatten(self, start_dim=0, end_dim=-1):
        # TODO: original, view or copy.
        self.tensor = self.tensor.flatten(start_dim, end_dim)
        self.mask = self.mask.flatten(1, 2)
        return self

    def to(self, device):
        return MaskedTensor.bind(
            *[getattr(self, attr).to(device) for attr in ("tensor", "mask")]
        )

    def detach(self):
        return MaskedTensor(
            *[getattr(self, attr).detach() for attr in ("tensor", "mask")]
        )

    def cpu(self):
        return MaskedTensor(
            *[getattr(self, attr).cpu() for attr in ("tensor", "mask")]
        )

    @staticmethod
    def _max_size(src: List[Tensor]):
        maxes = list(src[0].size())

        for tensor in src[1:]:
            for dim, size in enumerate(tensor.size()):
                maxes[dim] = max(size, maxes[dim])

        return Size(maxes)

    @staticmethod
    def _len_ones(tensor: Tensor):
        n = 0

        for x in tensor:
            if x == 1:
                n += 1
                continue
            if x == 0:
                return n
            raise TypeError

        return n
