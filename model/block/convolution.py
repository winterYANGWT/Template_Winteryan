import torch
import torch.nn as nn

__all__ = ['ChannelChanger', 'Conv3x3', 'DilatedConv2d']


class ChannelChanger(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_changer = nn.Conv2d(in_channels,
                                         out_channels,
                                         kernel_size=1)
        self.initialize()

    def forward(self, input_tensor):
        output_tensor = self.channel_changer(input_tensor)
        return output_tensor

    def initialize(self):
        nn.init.kaiming_normal_(self.channel_changer.weight)


class Conv3x3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              dilation=dilation,
                              groups=groups,
                              bias=bias,
                              padding_mode=padding_mode)
        self.initialize()

    def forward(self, input_tensor):
        output_tensor = self.conv(input_tensor)
        return output_tensor

    def initialize(self):
        nn.init.kaiming_normal_(self.conv.weight)


class DilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilated_ratios,
                 stride, padding):
        super().__init__()
        if not (isinstance(out_channels, (int, list))):
            msg = 'out_channels should be int or list of int.'
            raise ValueError(msg)

        for ratio in dilated_ratios:
            if not isinstance(ratio, int):
                msg = 'dilated_ratios should be list of int.'
                raise ValueError(msg)

        if isinstance(out_channels, int):
            out_channels_per_conv = out_channels // len(dilated_ratios)

            if out_channels_per_conv * len(dilated_ratios) != out_channels:
                msg = 'out_channels should be a multiple of len(dilated_ratios).'
                raise ValueError(msg)

            out_channels = [out_channels_per_conv] * len(dilated_ratios)
        else:
            if len(out_channels) != len(dilated_ratios):
                msg = 'len(out_channels) should be the same as len(dilated_ratios).'
                raise ValueError(msg)

            out_channels_per_conv = out_channels

        self.convs = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels_per_conv,
                      kernel_size=kernel_size,
                      stride=stride,
                      dilation=dilated_ratio_per_conv,
                      padding=padding + dilated_ratio_per_conv - 1)
            for (out_channels_per_conv,
                 dilated_ratio_per_conv) in zip(out_channels, dilated_ratios)
        ]
        self.convs = nn.ModuleList(self.convs)
        self.initialize()

    def forward(self, input_tensor):
        output_tensor = [conv(input_tensor) for conv in self.convs]
        output_tensor = torch.cat(output_tensor, dim=1)
        return output_tensor

    def initialize(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
