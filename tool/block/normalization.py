import torch
import torch.nn as nn

__all__ = ['AdaptiveInstanceNorm2d']


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_factor = 1e-10

    def forward(self, tensor, weight, bias):
        N, C, H, W = tensor.size()
        tensor = tensor.view(N, C, H * W)
        bias_in = torch.mean(tensor, dim=2, keepdim=True)
        weight_in = torch.std(tensor, dim=2, keepdim=True)
        out_tensor = (tensor - bias_in) / (weight_in +
                                           self.soft_factor) * weight + bias
        out_tensor = out_tensor.view(N, C, H, W)
        return out_tensor
