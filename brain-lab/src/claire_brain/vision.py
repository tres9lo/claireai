"""Small randomly initialized visual-grounding network for GUI screenshots."""

from __future__ import annotations

import torch
from torch import nn


class ScratchGrounder(nn.Module):
    """Predict a normalized click point from screenshot pixels and instruction tokens."""

    def __init__(self, *, vocab_size: int, width: int = 32, text_width: int = 64) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.token_embedding = nn.Embedding(vocab_size, text_width, padding_idx=257)
        self.head = nn.Sequential(
            nn.Linear(width * 4 * 4 * 4 + text_width, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        visual = self.image_encoder(image)
        mask = tokens.ne(257).unsqueeze(-1)
        embedded = self.token_embedding(tokens)
        text = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.head(torch.cat((visual, text), dim=1))
