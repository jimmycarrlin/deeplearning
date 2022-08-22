import math


def o(i: int, p: int, k: int, s: int):
    """Math from <https://arxiv.org/abs/1603.07285>."""
    return math.floor((i + 2 * p - k) / s) + 1