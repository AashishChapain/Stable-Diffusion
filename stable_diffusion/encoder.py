import torch
from torch import nn 
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__(
            # (batch_size, channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, h, w) -> (batch_size, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(128, 256),

            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/8, w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_AttentionBlock(512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            nn.GroupNorm(32, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            nn.SiLU(),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 8, h/8, w/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, h/8, w/8) -> (batch_size, 8, h/8, w/8)
            nn.Conv2d(8, 8, kernel_size=3, padding=1),

        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, h, w)
        # noise: (batch_size, out_channel, h/8, w/8)

        for module in self:
            # modules that include stride i.e. conv2d
            if getattr(module, 'stride', None) == (2, 2):
                # setting asymmetric padding i.e. on right and left of the image tensor
                # (left, right, top, bottom)
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)
        
        # (batch_size, 8, h/8, w/8) -> 2 tensors of shape = (batch_size, 4, h/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (batch_size, 4, h/8, w/8) -> # (batch_size, 4, h/8, w/8)
        log_variance = torch.clamp(log_variance, -30, 20) # clamping the value in range of -30 to 20

        # (batch_size, 4, h/8, w/8) -> # (batch_size, 4, h/8, w/8)
        variance = log_variance.exp()

        # # (batch_size, 4, h/8, w/8) -> # (batch_size, 4, h/8, w/8)
        std = variance.sqrt()

        # noise Z = N(0, 1) --> X = N(mean, variance)
        x = mean + std * noise

        # scale the output by a constant
        x *= 0.18215 # explained in their papaer

        return x

