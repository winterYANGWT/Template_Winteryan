import numpy as np
import torch


class OptimizerManager(object):
    config_template = {'lr': None, 'warmup': 0, 'grad_clip': -1, 'tick': 0}

    def __init__(self, scaler=None) -> None:
        super().__init__()
        self.scaler = scaler
        self.names = []
        self.optimizers = {}
        self.optimizers_config = {}

    def register(self,
                 optim_cls: torch.optim.Optimizer,
                 model: torch.nn.Module,
                 lr: float,
                 warmup: int = 0,
                 grad_clip: float = -1.,
                 **optim_params):
        name = model.name
        self.names.append(name)
        self.optimizers[name] = optim_cls(model.parameters(),
                                          lr=lr,
                                          **optim_params)
        self.optimizers_config[name] = OptimizerManager.config_template.copy()
        self.optimizers_config[name]['lr'] = lr
        self.optimizers_config[name]['warmup'] = warmup
        self.optimizers_config[name]['grad_clip'] = grad_clip

    def zero_grad(self, *names):
        process_names = names if len(names) != 0 else self.names

        for name in process_names:
            self.optimizers[name].zero_grad()

    def step(self, *names):
        def _step(name):
            warmup = self.optimizers_config[name]['warmup']
            tick = self.optimizers_config[name]['tick']
            lr = self.optimizers_config[name]['lr']
            grad_clip = self.optimizers_config[name]['grad_clip']

            if warmup > 0 and tick < warmup:
                for g in self.optimizers[name].param_groups:
                    g['lr'] = lr * np.minimum((tick + 1) / warmup, 1.)

            if grad_clip >= 0:
                for g in self.optimizers[name].param_groups:
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        g['params'], grad_clip)

            if self.scaler != None:
                self.scaler.step(self.optimizers[name])
            else:
                self.optimizers[name].step()

            self.optimizers_config[name]['tick'] += 1

        process_names = names if len(names) != 0 else self.names

        for name in process_names:
            _step(name)

        if self.scaler != None:
            self.scaler.update()
