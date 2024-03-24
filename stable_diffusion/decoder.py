import torch
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, h, w)
        residue = x
        n, c, h, w = x.shape

        # (batch_size, features, h, w) -> (batch_size, features, h * w)
        x = x.view(n, c, h * w)

        # (batch_size, features, h * w) -> (batch_size, , h * w, features)
        x = x.transpose(-1, -2)

        # (batch_size, features, h * w) -> (batch_size, , h * w, features)
        x = self.attention(x)

        # (batch_size, , h * w, features) -> (batch_size, features, h * w)
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            # it returns the tensor of same size as the input
            self.residual_layer = nn.Identity()
        else:
            # if the output size doesn't matches the size of the tensor that we're adding to, this block ensures that size of both tensor is same
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, h, w)
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)