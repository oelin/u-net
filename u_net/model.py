from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce


class ConvolutionBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.GroupNorm(
                num_groups=min(output_channels, 32),
                num_channels=output_channels,
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)


class DownsampleBlock(nn.Module):

    def __init__(self, input_channels) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.layers(x)


class UpsampleBlock(nn.Module):

    def __init__(self, input_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=input_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)


class DownBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: Optional[int] = None) -> None:
        super().__init__()

        output_channels = output_channels or input_channels * 2

        self.convolution_blocks = nn.Sequential(
            ConvolutionBlock(
                input_channels=input_channels,
                output_channels=output_channels,
            ),
            ConvolutionBlock(
                input_channels=output_channels,
                output_channels=output_channels,
            ),
        )

        self.downsample_block = DownsampleBlock(input_channels=output_channels)
        
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z = self.convolution_blocks(x)
        x = self.downsample_block(z)

        return x, z


class UpBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: Optional[int] = None) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            ConvolutionBlock(
                input_channels=input_channels * 2,
                output_channels=input_channels,
            ),
            ConvolutionBlock(
                input_channels=input_channels,
                output_channels=input_channels,
            ),
            ConvolutionBlock(
                input_channels=input_channels,
                output_channels=output_channels,
            ) if output_channels else UpsampleBlock(
                input_channels=input_channels,
            ),
        )
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        x = torch.cat((x, z), dim=-3)

        return self.layers(x)


class MiddleBlock(nn.Module):

    def __init__(self, input_channels: int) -> None:

        super().__init__()

        self.layers = nn.Sequential(
            ConvolutionBlock(
                input_channels=input_channels,
                output_channels=input_channels * 2,
            ),
            ConvolutionBlock(
                input_channels=input_channels * 2,
                output_channels=input_channels * 2,
            ),
            UpsampleBlock(input_channels=input_channels * 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layers(x)


@dataclass(frozen=True)
class UNetConfiguration:
    input_channels: int
    output_channels: int
    hidden_channels: int


class UNet(nn.Module):

    def __init__(self, configuration: UNetConfiguration) -> None:
        super().__init__()

        self.down1 = DownBlock(
            input_channels=configuration.input_channels,
            output_channels=configuration.hidden_channels,
        )

        self.down2 = DownBlock(
            input_channels=configuration.hidden_channels,
        )

        self.down3 = DownBlock(
            input_channels=configuration.hidden_channels * 2,
        )

        self.down4 = DownBlock(
            input_channels=configuration.hidden_channels * 4,
        )

        self.middle = MiddleBlock(
            input_channels=configuration.hidden_channels * 8,
        )

        self.up4 = UpBlock(
            input_channels=configuration.hidden_channels * 8,
        )

        self.up3 = UpBlock(
            input_channels=configuration.hidden_channels * 4,
        )

        self.up2 = UpBlock(
            input_channels=configuration.hidden_channels * 2,
        )

        self.up1 = UpBlock(
            input_channels=configuration.hidden_channels,
            output_channels=configuration.output_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x, z1 = self.down1(x)
        x, z2 = self.down2(x)
        x, z3 = self.down3(x)
        x, z4 = self.down4(x)
        
        x = self.middle(x)

        x = self.up4(x, z4)
        x = self.up3(x, z3)
        x = self.up2(x, z2)
        x = self.up1(x, z1)

        return x
