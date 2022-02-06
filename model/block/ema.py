import torch
import torch.nn as nn

__all__ = ['EMA']


class EMA(object):
    def __init__(self, beta) -> None:
        super().__init__()
        self.beta = beta

    def update_model(self, old_model: torch.Module, new_model: torch.Module):
        for old_params, new_params in zip(old_model.parameters(),
                                          new_model.parameters()):
            old_weight, new_weight = old_params.data, new_params.data
            new_params.data = self._update(old_weight, new_weight)

    def _update(self, old_weight, new_weight):
        if old_weight is None:
            return new_weight

        return old_weight * self.beta + new_weight * (1 - self.beta)
