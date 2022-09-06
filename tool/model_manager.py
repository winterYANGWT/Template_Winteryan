from typing import Dict, AnyStr, List
import torch
from utils import to_device
from collections import UserDict


class ModelManager(UserDict):
    def __init__(self) -> None:
        super().__init__()
        self.step = 0
        self.device = torch.device('cpu')
        self.is_data_parallel = False

    def registe(self, model, name=None):
        name = name if name != None else model.name
        self.data[name] = model

    def data_parallel(self):
        for name in self.data.keys():
            if not isinstance(self.data[name], torch.nn.DataParallel):
                self.data[name] = torch.nn.DataParallel(self.data[name])

                for registed_func in self.data[name].module.registed_funcs:
                    setattr(self.data[name], registed_func,
                            getattr(self.data[name].module, registed_func))

    def train(self):
        for name in self.keys():
            self.data[name].train()

    def eval(self):
        for name in self.keys():
            self.data[name].eval()

    def to_device(self, device):
        self.device = device
        self.data = to_device(self.data, device)

    def get_parameters(self, name=None):
        def _get_parameters(name):
            model = self.data[name]

            if not isinstance(model, torch.nn.DataParallel):
                parameters = model.parameters()
            else:
                parameters = model.module.parameters()

            return parameters

        parameters_dict = {}

        if name != None:
            parameters_dict[name] = _get_parameters(name)
        else:
            for name in self.keys():
                parameters_dict[name] = _get_parameters(name)

        return parameters_dict

    def set_parameters(self, parameters_dict: Dict[AnyStr,
                                                   List[torch.nn.Parameter]]):
        for name in parameters_dict.keys():
            model_parameters = self.get_parameters(name)

            for new_params, old_params in zip(parameters_dict[name],
                                              model_parameters[name]):
                device = old_params.device
                dtype = old_params.dtype
                _new_params = new_params.to(device=device, dtype=dtype)
                old_params.copy_(_new_params.data)

    def state_dict(self):
        state_dict = {}

        for name in self.keys():
            model = self.data[name]

            if isinstance(model, torch.nn.DataParallel):
                model = model.module.cpu()
            else:
                model = model.cpu()

            state_dict[name] = model.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict):
        for name in self.keys:
            if name in state_dict.keys():
                self.data[name].load_state_dict(state_dict[name])
