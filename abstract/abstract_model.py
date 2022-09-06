import torch.nn as nn

__all__ = ['AbstractModel']


class AbstractModel(nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name
        self.registed_funcs = ['name']

    def registe(self, func):
        name = func.__name__

        if name not in self.registed_funcs:
            self.registed_funcs.append(name)

        return func
