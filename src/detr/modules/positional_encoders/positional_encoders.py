from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor


class PositionalEncoder(nn.Module):
    pass


class RelativeLearned2DPE(nn.Module):
    """Relative Learned Positional Encoder. Bello et al <https://arxiv.org/abs/1904.09925>."""
    # TODO: Make more generic
    def __init__(self, d: int, W_max: int, H_max: int):
        super().__init__()

        self.W_max = W_max
        self.H_max = H_max

        self.rel_embed_h = nn.Parameter(data=torch.randn(2*H_max - 1, d) + d**-0.5)
        self.rel_embed_w = nn.Parameter(data=torch.randn(2*W_max - 1, d) + d**-0.5)

    def forward(self, q: Tensor):
        _, H, W, _ = q.size()
        W_max, H_max = self.max_size

        pos_logits_w = self._rel_logits_1d(
            q,
            self.rel_embed_w[(W_max - W):(W_max + W - 1), :],
            (0, 1, 3, 2, 4)
        )
        pos_logits_h = self._rel_logits_1d(
            torch.transpose(q, 1, 2),
            self.rel_embed_h[(H_max - H):(H_max + H - 1), :],
            (0, 3, 1, 4, 2)
        )
        pos_logits = pos_logits_h + pos_logits_w

        return pos_logits

    @property
    def max_size(self):
        return self.W_max, self.H_max

    def _rel_logits_1d(self, q: Tensor, rel_k: Tensor, transpose_mask: Tuple[int, ...]):
        """Compute relative logits along one dimension."""
        _, H, _, _ = q.size()

        # Out: (B, H, W, 2*W - 1)
        rel_logits = torch.einsum("bxyd,md->bxym", q, rel_k)

        # Out: (B, H, W, W)
        rel_logits = self._rel_to_abs(rel_logits)

        # Out: (B, H, 1, W, W)
        rel_logits = torch.unsqueeze(rel_logits, dim=2)

        # Out: (B, H, H, W, W)
        rel_logits = torch.tile(rel_logits, dims=(1, 1, H, 1, 1))

        # Out: (B, H, W, H, W)
        rel_logits = torch.permute(rel_logits, transpose_mask)

        return rel_logits

    @staticmethod
    def _rel_to_abs(x: Tensor):
        """Converts tensor from relative to absolute indexing."""
        # (B, X, Y, 2Y-1)
        B, X, Y, _ = x.size()

        x = torch.cat((x, torch.zeros(B, X, Y, 1, device=x.device)), dim=3)
        x = torch.reshape(x, (B, X, Y*2*Y))

        x = torch.cat((x, torch.zeros(B, X, Y-1, device=x.device)), dim=2)
        x = torch.reshape(x, (B, X, Y+1, 2*Y - 1))

        x = x[:, :, :Y, Y-1:]

        return x


class AbsoluteLearned2DPE(nn.Module):
    """Absolute Learned Positional Encoder."""

    def __init__(self, d: int, W_max: int, H_max: int):
        super().__init__()

        self.W_max = W_max
        self.H_max = H_max

        self.embed_h = nn.Parameter(data=torch.randn(H_max, d) + d**-0.5)
        self.embed_w = nn.Parameter(data=torch.randn(W_max, d) + d**-0.5)

    def forward(self, q: Tensor, k: Tensor):
        # Out: (H, W, d)
        embed = self.embed_h.unsqueeze(1) + self.embed_w.unsqueeze(0)

        # Out: (B, Nh, H, W, H, W)
        logits = _logits_from_queries_and_keys(q, embed, k, embed)

        return logits

    @property
    def max_size(self):
        return self.W_max, self.H_max


class AbsoluteSinusoidal2DPE(nn.Module):
    """Absolute Sinusoidal Positional Encoder."""

    def __init__(self, d: int, W_max: int, H_max: int):
        super().__init__()

        self.W_max = W_max
        self.H_max = H_max

        # Out: (H or W, d)
        embed_h = self._embeds_1d(H_max, d)
        embed_w = self._embeds_1d(W_max, d)

        # Out: (H, W, d)
        self.embed = embed_h.unsqueeze(1) + embed_w.unsqueeze(0)

    def forward(self, q: Tensor, k: Tensor):
        embed = self.embed
        logits = _logits_from_queries_and_keys(q, embed, k, embed)

        return logits

    @staticmethod
    def _embeds_1d(L: int, dkh: int):
        numer = torch.arange(L)
        denom = torch.full((L, dkh), L) ** (2 * torch.ones(L, dkh).cumsum(dim=1) / dkh)

        embed = numer / denom
        embed[:, 0::2] = embed[:, 0::2].sin()
        embed[:, 1::2] = embed[:, 1::2].cos()

        return embed

    @property
    def max_size(self):
        return self.W_max, self.H_max


class AbsoluteLearnedPE(nn.Module):
    """Absolute Learned Positional Encoder."""

    def __init__(self, d: int, L_max: int):
        super().__init__()

        self.L_max = L_max
        self.embed = nn.Parameter(data=torch.randn(L_max, d) + d**-0.5)

    def forward(self, q: Tensor, k: Tensor):
        _, Q, _ = q.size()
        _, K, _ = k.size()

        # Out: (Q or K, d)
        q_embed = self.embed[:Q, :]
        k_embed = self.embed[:K, :]

        # Out: (B, Q, K)
        logits = _logits_from_queries_and_keys(q, k, q_embed, k_embed)

        return logits

    @property
    def max_size(self):
        return self.L_max


class AbsoluteSinusoidalPE(nn.Module):
    """Absolute Sinusoidal Positional Encoder."""

    def __init__(self, d: int, L_max: int):
        super().__init__()

        self.L_max = L_max
        self.embed = self._embeds_1d(L_max, d)

    def forward(self, q: Tensor, k: Tensor):
        _, _, Q, _ = q.size()
        _, _, K, _ = k.size()

        # Out: (Q or K, d)
        q_embed = self.embed[:Q, :]
        k_embed = self.embed[:K, :]

        # Out: (B, Q, K)
        logits = _logits_from_queries_and_keys(q, k, q_embed, k_embed)

        return logits

    @property
    def max_size(self):
        return self.L_max

    @staticmethod
    def _embeds_1d(L: int, dkh: int):
        numer = torch.arange(L)
        denom = torch.full((L, dkh), L) ** (2 * torch.ones(L, dkh).cumsum(dim=1) / dkh)

        embed = numer / denom
        embed[:, 0::2] = embed[:, 0::2].sin()
        embed[:, 1::2] = embed[:, 1::2].cos()

        return embed


def _logits_from_queries_and_keys(q: Tensor, k: Tensor, q_embed: Tensor, k_embed: Tensor):
    if all([q.dim() == 3, k.dim() == 3, q_embed.dim() == 2, k_embed.dim() == 2]):
        return _logits_from_queries_and_keys1d(q, k, q_embed, k_embed)
    if all([q.dim() == 4, k.dim() == 4, q_embed.dim() == 3, k_embed.dim() == 3]):
        return _logits_from_queries_and_keys2d(q, k, q_embed, k_embed)
    raise RuntimeError("Bad dims.")


def _logits_from_queries_and_keys1d(q: Tensor, k: Tensor, q_embed: Tensor, k_embed: Tensor):
    return torch.einsum("bqd,kd->bqk", q, k_embed) + torch.einsum("qd,bkd->bqk", q_embed, k + k_embed)


def _logits_from_queries_and_keys2d(q: Tensor, k: Tensor, q_embed: Tensor, k_embed: Tensor):
    return torch.einsum("bijd,kld->bijkl", q, k_embed) + torch.einsum("ijd,bkld->bijkl", q_embed, k + k_embed)


def _get_pos_encoder(pos_encoder: str, d: int, *max_size: int):
    if pos_encoder == "rl2d":
        return RelativeLearned2DPE(d, *max_size)
    if pos_encoder == "al2d":
        return AbsoluteLearned2DPE(d, *max_size)
    if pos_encoder == "al1d":
        return AbsoluteLearnedPE(d, *max_size)
    if pos_encoder == "as1d":
        return AbsoluteSinusoidalPE(d, *max_size)
    raise RuntimeError("No such positional encoder.")
