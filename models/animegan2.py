"""AnimeGANv2 Generator compatible with bryandlee/animegan2-pytorch.

The module names and forward pass follow the public PyTorch implementation:
https://github.com/bryandlee/animegan2-pytorch/blob/main/model.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormLReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding=1,
        pad_mode: str = "reflect",
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise ValueError(f"Unsupported padding mode: {pad_mode}")
        super().__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        self.use_res_connect = in_channels == out_channels
        bottleneck = int(round(in_channels * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(
                ConvNormLReLU(in_channels, bottleneck, kernel_size=1, padding=0)
            )
        layers.append(
            ConvNormLReLU(
                bottleneck,
                bottleneck,
                groups=bottleneck,
                bias=True,
            )
        )
        layers.append(nn.Conv2d(bottleneck, out_channels, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels, affine=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.layers(input_tensor)
        if self.use_res_connect:
            output = input_tensor + output
        return output


class Generator(nn.Module):
    """AnimeGANv2 image generator; it accepts BCHW tensors in [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64),
        )
        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128),
        )
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128),
        )
        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3),
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input_tensor: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
        output = self.block_a(input_tensor)
        half_size = output.size()[-2:]
        output = self.block_b(output)
        output = self.block_c(output)

        if align_corners:
            output = F.interpolate(output, half_size, mode="bilinear", align_corners=True)
        else:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.block_d(output)
        if align_corners:
            output = F.interpolate(
                output,
                input_tensor.size()[-2:],
                mode="bilinear",
                align_corners=True,
            )
        else:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.block_e(output)
        return self.out_layer(output)


__all__ = ["Generator"]

