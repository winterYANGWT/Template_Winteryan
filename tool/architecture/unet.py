import torch
import torch.nn as nn
from ..block import SubPixelUpSample, ConvDownSample
from functools import partial

__all__ = ['UNet']


class UNetOneLevel(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 left_block,
                 right_block,
                 next_level,
                 down=None,
                 up=None,
                 **kwargs) -> None:
        super().__init__()
        self.level = next_level.level + 1
        self.left = left_block(in_channels=in_channels,
                               out_channels=mid_channels,
                               level=self.level,
                               **kwargs)
        self.right = right_block(in_channels=2 * mid_channels,
                                 out_channels=out_channels,
                                 level=self.level,
                                 **kwargs)
        self.next_level = next_level

        try:
            self.down = down(in_channels=mid_channels, kernel_size=2)
            self.up = up(in_channels=mid_channels, kernel_size=2)
        except TypeError:
            self.down = down(kernel_size=2)
            self.up = up(kernel_size=2)

    def forward(self, in_tensor, **kwargs):
        left_out_tensor = self.left(in_tensor, **kwargs)
        next_level_in_tensor = self.down(left_out_tensor)
        next_level_out_tensor = self.next_level(next_level_in_tensor, **kwargs)
        next_level_out_tensor = self.up(next_level_out_tensor)
        right_in_tensor = torch.cat([left_out_tensor, next_level_out_tensor],
                                    dim=-3)
        right_out_tensor = self.right(right_in_tensor, **kwargs)
        return right_out_tensor


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels_list,
                 left_block,
                 mid_block,
                 right_block,
                 predict_block,
                 down_sample=None,
                 up_sample=None,
                 **kwargs) -> None:
        super().__init__()
        down_sample = down_sample if down_sample != None else ConvDownSample
        up_sample = up_sample if up_sample != None else SubPixelUpSample
        in_out_list = list(zip(mid_channels_list[:-1], mid_channels_list[1:]))
        r_in_out_list = reversed(in_out_list)
        self.net = mid_block(in_channels=mid_channels_list[-1],
                             out_channels=mid_channels_list[-1],
                             level=0,
                             **kwargs)
        self.net.level = 0

        _UNetOneLevel = partial(UNetOneLevel,
                                left_block=left_block,
                                right_block=right_block,
                                down=down_sample,
                                up=up_sample)

        for in_ch, out_ch in r_in_out_list:
            self.net = _UNetOneLevel(in_channels=in_ch,
                                     mid_channels=out_ch,
                                     out_channels=in_ch,
                                     next_level=self.net,
                                     **kwargs)

        self.net = _UNetOneLevel(in_channels=in_channels,
                                 mid_channels=mid_channels_list[0],
                                 out_channels=mid_channels_list[0],
                                 next_level=self.net,
                                 **kwargs)

        self.out = predict_block(in_channels=mid_channels_list[0],
                                 out_channels=out_channels,
                                 **kwargs)

    def forward(self, in_tensor, **kwargs):
        out_tensor = self.out(self.net(in_tensor, **kwargs), **kwargs)
        return out_tensor
