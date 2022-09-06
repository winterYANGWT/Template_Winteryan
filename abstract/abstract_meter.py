from abc import ABC, abstractmethod

__all__ = ['AbstractMeter']


class AbstractMeter(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.sum = {}
        self.num_samples = -1
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, input_data, output_data, loss):
        pass

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
