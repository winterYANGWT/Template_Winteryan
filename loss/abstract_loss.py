import torch.nn as nn
from abc import abstractclassmethod


class AbstractLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def forward(self):
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)
