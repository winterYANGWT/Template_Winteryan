from collections import UserDict
from abstract import AbstractMeter
from .global_info import GlobalInfo


class MeterDict(UserDict):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        l = [str(v) for v in self.values()]
        s = ', '.join(l)
        return s

    def __repr__(self) -> str:
        return str(self)


class LossMeter(AbstractMeter):
    def __init__(self, name, mode='mean') -> None:
        self.name = name
        super().__init__()

        if mode not in ['mean', 'sum', 'none']:
            msg = 'mode should be mean, sum or none.'
            raise ValueError(msg)
        else:
            self.mode = mode

    def reset(self):
        self.sum = {}
        self.sum[self.name] = -1
        self.num_samples = 0

    def update(self, input_data, output_data, loss):
        _loss = loss[self.name].item()
        num_samples = loss['num_' + self.name]

        if self.mode == 'mean':
            self.sum[self.name] += _loss * num_samples
        elif self.mode == 'sum':
            self.sum[self.name] += _loss
        else:
            self.sum[self.name] += _loss.sum()

        self.num_samples += num_samples


class MeterManager(object):
    def __init__(self) -> None:
        super().__init__()
        self.train_meters = MeterDict()
        self.eval_meters = MeterDict()

    def initialize(self):
        for meter in self.train_meters.values():
            meter.reset()

        for meter in self.eval_meters.values():
            meter.reset()

    def register(self, name, meter, mode):
        if mode == 'train':
            self.train_meters[name] = meter
        elif mode == 'eval':
            self.eval_meters[name] = meter

    def update(self, input_data, output_data, loss):
        if GlobalInfo.mode == 'train':
            meters = self.train_meters
        elif GlobalInfo.mode == 'eval':
            meters = self.eval_meters

        for meter in meters.values():
            meter.update(input_data, output_data, loss)

    def __str__(self):
        if GlobalInfo.mode == 'train':
            return str(self.train_meters)
        elif GlobalInfo.mode == 'eval':
            return str(self.eval_meters)

    def __repr__(self) -> str:
        return str(self)
