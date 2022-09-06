from typing import Tuple
import torch

__all__ = ['Scale', 'Unscale']


class Scale(object):
    valid_check_rules = [
        lambda target_range: isinstance(target_range, (tuple, list)),
        lambda target_range: len(target_range) >= 2,
        lambda target_range: isinstance(target_range[0],
                                        (int, float)) and isinstance(
                                            target_range[1], (int, float)),
        lambda target_range: target_range[0] < target_range[1]
    ]

    def __init__(self, target_range: Tuple[float, float]) -> None:
        '''
        Args:
            target_range: (min,max)
        '''
        super().__init__()

        for rule in Scale.valid_check_rules:
            if rule(target_range) == False:
                msg = 'taget_range should be a tuple or a list whose length is equal or greater than 2, and its values are int or float. target_range[0]<target_range[1] (min,max).'
                raise ValueError(msg)

        self.min = target_range[0]
        self.max = target_range[1]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.mul_(self.max - self.min)
        tensor.add_(self.min)
        return tensor


class Unscale(object):
    valid_check_rules = [
        lambda original_range: isinstance(original_range, (tuple, list)),
        lambda original_range: len(original_range) >= 2,
        lambda original_range: isinstance(original_range[0],
                                          (int, float)) and isinstance(
                                              original_range[1], (int, float)),
        lambda original_range: original_range[0] < original_range[1]
    ]

    def __init__(self, original_range: Tuple[float, float]) -> None:
        '''
        Args:
            target_range: (min,max)
        '''
        super().__init__()

        for rule in Scale.valid_check_rules:
            if rule(original_range) == False:
                msg = 'original_range should be a tuple or a list whose length is equal or greater than 2, and its values are int or float. original_range[0]<original_range[1] (min,max).'
                raise ValueError(msg)

        self.min = original_range[0]
        self.max = original_range[1]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor.sub_(self.min)
        tensor.div_(self.max - self.min)
        return tensor
