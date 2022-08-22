import torch
from torch import Tensor


def batched_cartesian_product(tensor1: Tensor, tensor2: Tensor):
    return torch.tensor([torch.cartesian_prod(t1, t2).tolist() for t1, t2 in zip(tensor1, tensor2)]).to(tensor1.device)
