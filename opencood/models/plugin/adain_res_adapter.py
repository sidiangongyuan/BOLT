"""
Camera → LiDAR feature plugin for base-free heterogeneous collaboration.

Design goals:
  - Stabilize scale/statistics mismatch via AdaIN (conditioned on ego LiDAR stats)
  - Provide non-linear channel mixing via a lightweight residual CNN adapter
  - Keep parameters small and TTT-friendly (few-shot, unlabeled adaptation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def _channel_mean_std(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean/std over spatial dims.

    Args:
        x: [B, C, H, W]
    Returns:
        mean: [B, C, 1, 1]
        std:  [B, C, 1, 1]
    """
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    return mean, std


class _ResBlock2d(nn.Module):
    def __init__(self, channels: int, gn_groups: int):
        super().__init__()
        groups = min(gn_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return self.act(x + residual)


@dataclass(frozen=True)
class AdaINResAdapterConfig:
    in_channels: int = 64
    hidden_channels: int = 128
    num_blocks: int = 3
    gn_groups: int = 16
    adain_eps: float = 1e-5
    # Learnable mixing between raw feature x and AdaIN-normalized feature x_adain:
    #   x_base = x + sigmoid(alpha) * (x_adain - x)
    # Set a large positive init logit to recover previous behavior (almost full AdaIN).
    # Set a large negative init logit for identity-first behavior (almost bypass AdaIN).
    adain_alpha_init_logit: float = 10.0
    gate_init_logit: float = 0.0


class AdaINResAdapterPlugin(nn.Module):
    """
    AdaIN + residual CNN adapter + per-channel gate (on residual delta).

    Forward signature intentionally takes both:
      - x:   neighbor (camera) feature
      - ref: ego (LiDAR) feature as statistic reference
    """

    def __init__(self, cfg: AdaINResAdapterConfig):
        super().__init__()
        self.cfg = cfg

        in_ch = cfg.in_channels
        hid = cfg.hidden_channels

        self.adain_alpha_logits = nn.Parameter(
            torch.full((in_ch,), float(cfg.adain_alpha_init_logit))
        )

        self.pre = nn.Conv2d(in_ch, hid, kernel_size=1, bias=False)
        self.pre_gn = nn.GroupNorm(min(cfg.gn_groups, hid), hid)
        self.act = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(*[_ResBlock2d(hid, cfg.gn_groups) for _ in range(cfg.num_blocks)])

        self.post = nn.Conv2d(hid, in_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

        self.gate_logits = nn.Parameter(torch.full((in_ch,), float(cfg.gate_init_logit)))

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, C, H, W] camera feature
            ref: [B, C, H, W] ego LiDAR feature (same BEV shape)
        Returns:
            adapted x with same shape as input.
        """
        if x.ndim != 4 or ref.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got x={x.shape}, ref={ref.shape}")
        if x.shape != ref.shape:
            raise ValueError(f"Shape mismatch: x={x.shape}, ref={ref.shape}")

        mean_x, std_x = _channel_mean_std(x, self.cfg.adain_eps)
        mean_r, std_r = _channel_mean_std(ref, self.cfg.adain_eps)
        x_adain = (x - mean_x) / std_x * std_r + mean_r

        alpha = torch.sigmoid(self.adain_alpha_logits).view(1, -1, 1, 1)
        x_base = x + (x_adain - x) * alpha

        h = self.pre(x_base)
        h = self.pre_gn(h)
        h = self.act(h)
        h = self.blocks(h)
        delta = self.post(h)

        gate = torch.sigmoid(self.gate_logits).view(1, -1, 1, 1)
        return x_base + delta * gate
