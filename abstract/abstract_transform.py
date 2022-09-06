from typing import Any, Dict
from abc import ABC

__all__ = ['AbstractTransform']


class AbstractTransform(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.transforms = {}

    def __call__(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = {}

        for key in input_dict.keys():
            if key in self.transforms.keys():
                output_dict[key] = self.transforms[key](input_dict[key])
            else:
                output_dict[key] = input_dict[key]

        return output_dict
