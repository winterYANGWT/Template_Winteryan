from abc import ABC
import torch
from torch.cuda.amp import autocast, GradScaler
import utils
import meter

__all__ = ['TrainRunner', 'InferRunner', 'TestRunner']


class TrainRunner(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = GradScaler()
        self.datasets = {}
        self.data_iterators = {}
        self.models = {}
        self.criterions = {}
        self.optimizers = {}
        self.meters = {'train': meter.MeterDict(), 'val': meter.MeterDict()}

        self.load_config()
        self.load_datasets()
        self.load_models()
        self.load_optimizers()
        self.load_criterions()
        self.load_meters()

    def load_config(self):
        '''
        Load config from config package.

        Example:
            >>> self.cfg = UserDefinedConfig()
        '''
        msg = 'load_config should be implemented by subclass.'
        raise NotImplementedError(msg)

    def load_datasets(self):
        '''
        Load dataset(s) from dataset package.

        Example:
            >>> datasets = {}
            >>> datasets['train'] = A()
            >>> datasets['val'] = B()
            >>> self.data_loaders = utils.generate_data_loaders(datasets)
        '''
        msg = 'load_datasets should be implemented by subclass.'
        raise NotImplementedError(msg)

    def load_models(self):
        '''
        Load model(s) from model package.

        Example:
            >>> self.models = {}
            >>> self.models['a'] = A()
            >>> self.models['b'] = B()
            >>> self.models = utils.generate_models(self.models,
            >>>                                     self.cfg.IS_LOAD,
            >>>                                     self.cfg.DEVICE,
            >>>                                     self.cfg.DATA_PARALLEL)
        '''
        msg = 'load_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    def load_optimizers(self):
        '''
        Load optimizer(s) for model(s).
        '''
        msg = 'load_optimizers should be implemented by subclass.'
        raise NotImplementedError(msg)

    def load_criterions(self):
        '''
        Load criterion(s) from loss package.
        '''
        msg = 'load_criterions should be implemented by subclass.'
        raise NotImplementedError(msg)

    def load_meters(self):
        '''
        Load meter(s) from meter package.
        '''
        msg = 'load_meters should be implemented by subclass.'
        raise NotImplementedError(msg)

    def forward(self, input_data):
        '''
        Forward inference.
        '''
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)

    def compute_loss(self, input_data, output_data):
        '''
        Compute loss.
        '''
        msg = 'compute_loss should be implemented by subclass.'
        raise NotImplementedError(msg)

    def backward(self, loss, is_optimize):
        '''
        Backward propagation and update parameters.
        '''
        msg = 'backward should be implemented by subclass.'
        raise NotImplementedError(msg)

    def train(self, step, initial_flag):
        '''
        Train phase.
        '''
        if initial_flag == True:
            utils.initialize_meters(self.meters)
            utils.initialize_optimizers(self.optimizers)

            for key in self.models.keys():
                self.models[key].train()

        try:
            input_data = next(self.data_iterators['train'])
        except (StopIteration, KeyError):
            self.data_iterators['train'] = iter(self.data_loaders['train'])
            input_data = next(self.data_iterators['train'])

        input_data['phase'] = 'train'
        input_data = utils.to_device(input_data, self.cfg.DEVICE)

        with autocast():
            output_data = self.forward(input_data)
            loss = self.compute_loss(input_data, output_data)

        is_optimize = step % self.cfg.GRAD_ACCUMULATE_EVERY == 0
        self.backward(loss, is_optimize)

        if is_optimize == True:
            self.scaler.update()
            utils.initialize_optimizers(self.optimizers)

        self.update_meters(input_data, output_data, loss)

    @torch.no_grad()
    def val(self, step):
        '''
        Validate phase.
        '''
        msg = 'val should be implemented by subclass.'
        raise NotImplementedError(msg)

    def save_models(self, save_dir):
        '''
        Save model(s).
        '''
        for value in self.models.values():
            utils.save_model(value, save_dir)

    def update_meters(self, input_data, output_data, loss):
        '''
        Update meter(s).
        '''
        msg = 'update_meters should be implemented by subclass.'
        raise NotImplementedError(msg)


class InferRunner(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.load_config()
        self.load_transforms()
        self.load_models()

    @classmethod
    def load_config(self):
        '''
        Load config from config package.
        '''
        msg = 'load_config should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_transforms(self):
        '''
        Load transform(s) from transform package.
        '''
        msg = 'load_transforms should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_models(self):
        '''
        Load model(s) from model package.
        '''
        msg = 'load_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def get_data(self):
        '''
        Get data from user's input.
        '''
        msg = 'get_data should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def forward(self, input_data):
        '''
        Forward inference.
        '''
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def save_results(self, input_data, output_data):
        '''
        Save results.
        '''
        msg = 'save_results should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def infer(self):
        '''
        Inference phase.
        '''
        msg = 'infer should be implemented by subclass.'
        raise NotImplementedError(msg)


class TestRunner(ABC):
    def __init__(self) -> None:
        super().__init__()
