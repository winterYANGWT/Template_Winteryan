from abstract import AbstractMeter


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
        self.sum[self.name] = 0
        self.num_samples = 0

    def update(self, loss, num_samples):
        if self.mode == 'mean':
            self.sum[self.name] += loss * num_samples
        elif self.mode == 'sum':
            self.sum[self.name] += loss
        else:
            self.sum[self.name] += loss.sum()

        self.num_samples += num_samples
