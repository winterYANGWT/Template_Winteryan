from typing import Any, Dict
from abc import ABC, abstractclassmethod

__all__ = ['AbstractTransform']


class AbstractTransform(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractclassmethod
    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        msg = '__call__ should be implemented by subclass.'
        raise NotImplementedError(msg)
