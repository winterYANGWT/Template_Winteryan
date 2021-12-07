import torch
import torch.nn as nn

__all__ = ['ACE']


class ACE(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.soft_factor = 1e-10

        if reduction not in ['none', 'mean', 'sum']:
            msg = 'reduction should be none, mean or sum.'
            raise ValueError(msg)
        else:
            self.reduction = reduction

    def forward(self, probs, targets):
        probs += self.soft_factor
        N, T, K = probs.size()
        aggregation_probs = torch.sum(probs, dim=1)
        counted_targets = []

        for target in targets:
            placeholder = torch.zeros(size=(K),
                                      dtype=aggregation_probs.dtype,
                                      device=aggregation_probs.device)
            count = torch.bincount(target)
            placeholder[:count.size(0)] = count
            counted_targets.append(placeholder)

        counted_targets = torch.stack(counted_targets)
        mean_probs = aggregation_probs / T
        mean_targets = counted_targets / T
        loss = -torch.sum(torch.log(mean_probs) * mean_targets, dim=1)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            loss = loss.mean()
            return loss
        else:
            loss = loss.sum()
            return loss
