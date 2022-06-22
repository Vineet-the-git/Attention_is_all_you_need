from turtle import forward
from typing_extensions import Self
from requests import head
import torch
import torch.nn as nn
import torch.nn.functional as F
from self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, k, heads) -> None:
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )


    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x2 = self.ff(x)
        return self.norm2(x + x2)