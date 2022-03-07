from abc import ABC, abstractclassmethod


class AbstractMeter(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        msg = 'reset should be implemented by subclass.'
        raise NotImplementedError(msg)

    def update(self):
        msg = 'update should be implemented by subclass.'
        raise NotImplementedError(msg)

    def compute_value(self):
        value = {}

        for key in self.sum.keys():
            if self.num_samples != 0:
                value[key] = self.sum[key] / self.num_samples
            else:
                value[key] = 0

        return value

    @property
    def value(self) -> dict:
        return self.compute_value()

    def __str__(self):
        value_dict = self.compute_value()
        items = [f'{name}: {value:.6}' for (name, value) in value_dict.items()]
        return ', '.join(items)
