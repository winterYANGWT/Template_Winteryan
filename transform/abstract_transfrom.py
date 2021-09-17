from typing import Dict
import torch
from abc import ABC, abstractclassmethod


class AbstractTransform(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def __call__(self, input_tensor_dict: Dict[str,
                                               torch.Tensor]) -> torch.Tensor:
        msg = '__call__ should be implemented by subclass.'
        raise NotImplementedError(msg)
