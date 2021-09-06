import torch.nn as nn


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
