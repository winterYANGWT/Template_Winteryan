import torch.nn as nn
from .convolution import ChannelChanger

__all__ = ['DeconvUpSample', 'SubPixelUpSample', 'ConvDownSample']


# up sample
class DeconvUpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=2):
        super().__init__()
        self.bicubic = nn.Upsample(scale_factor=kernel_size)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)

    def forward(self, input_tensor):
        output_tensor = self.bicubic(input_tensor) + self.bicubic(
            self.conv(input_tensor))
        return output_tensor


class SubPixelUpSample(nn.Module):
    def __init__(self, in_channels, kernel_size=2):
        super().__init__()
        out_channels = in_channels * kernel_size**2
        self.expand_channel = ChannelChanger(in_channels=in_channels,
                                             out_channels=out_channels)
        self.pixel_shuffle = nn.PixelShuffle(kernel_size)

    def forward(self, input_tensor):
        output_tensor = self.pixel_shuffle(self.expand_channel(input_tensor))
        return output_tensor


# down sample
class ConvDownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=2):
        super().__init__()
        padding = kernel_size // 2
        conv_kernel_size = padding * 2 + 1
        self.down_sample = nn.Conv2d(in_channels=in_channels,
                                     out_channels=in_channels,
                                     kernel_size=conv_kernel_size,
                                     stride=kernel_size,
                                     padding=padding)

    def forward(self, input_tensor):
        output_tensor = self.down_sample(input_tensor)
        return output_tensor
