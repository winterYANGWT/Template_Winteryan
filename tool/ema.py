import torch.nn as nn
import contextlib
from typing import Union, Dict
from .model_manager import ModelManager
from .global_info import GlobalInfo


class ExponentialMovingAverage(object):
    def __init__(self,
                 beta=0.995,
                 use_ema=False,
                 start_step=1,
                 update_every=1) -> None:
        super().__init__()
        self.use_ema = use_ema
        self.start_step = start_step
        self.update_every = update_every

        if beta < 0 or beta > 1:
            msg = f'beta must be between 0 and 1, but got {beta}.'
            raise ValueError(msg)
        else:
            self.beta = beta

        self.shadow_parameters_dict = {}
        self.store_parameters_dict = {}
        self.num_updates = 0
        self.running = False

    def registe(self, models: ModelManager):
        parameters_dict = models.get_parameters()

        for name in parameters_dict.keys():
            self.shadow_parameters_dict[name] = [
                p.cpu().clone().detach() for p in parameters_dict[name]
            ]

    def update(self, models: ModelManager):
        self.num_updates += 1
        parameters_dict = models.get_parameters()

        for name in self.shadow_parameters_dict.keys():
            new_params = [
                p.cpu().clone().detach() for p in parameters_dict[name]
            ]
            beta = min(self.beta,
                       (1 + self.num_updates) / (10 + self.num_updates))

            for shadow_param, new_param in zip(
                    self.shadow_parameters_dict[name], new_params):
                tmp_param = shadow_param - new_param
                tmp_param.mul_(1 - beta)
                shadow_param.sub_(tmp_param)

    def store(self, models: ModelManager):
        parameters_dict = models.get_parameters()

        for name in self.store_parameters_dict.keys():
            self.store_parameters_dict[name] = [
                p.clone() for p in parameters_dict[name]
            ]

    def restore(self, models: ModelManager):
        models.set_parameters(self.store_parameters_dict)

    def copy(self, models: ModelManager):
        models.set_parameters(self.shadow_parameters_dict)

    @contextlib.contextmanager
    def apply_ema(self, models: Union[nn.Module, Dict[str, nn.Module]]):
        if self.running:
            self.store(models)
            self.copy(models)

            try:
                yield
            finally:
                self.restore(models)
        else:
            yield

    def state_dict(self):
        state_dict = {
            'beta': self.beta,
            'num_updates': self.num_updates,
            'shadow': self.shadow_parameters_dict
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        self.beta = state_dict['beta']
        self.num_updates = state_dict['num_updates']
        self.shadow_parameters_dict = state_dict['shadow']

    def __getitem__(self, key):
        return self.shadow_parameters_dict[key]

    def keys(self):
        return self.shadow_parameters_dict.keys()

    def values(self):
        return self.shadow_parameters_dict.values()

    def items(self):
        return self.shadow_parameters_dict.items()

    def step(self, model_manager: ModelManager):
        if self.use_ema == True:
            step = GlobalInfo.step

            if step >= self.start_step:
                self.running = True
            else:
                self.running = False

            if step == self.start_step:
                self.registe(model_manager)

            if step > self.start_step and step % self.update_every == 0:
                self.update(model_manager)
