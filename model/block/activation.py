from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Mish']


class Mish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor * torch.tanh(F.softplus(input_tensor))
        return output_tensor
