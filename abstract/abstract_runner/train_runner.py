from abc import ABC, abstractmethod
import torch
from torch.cuda.amp import autocast, GradScaler
import os.path as path
from tqdm import tqdm
from tool import ExponentialMovingAverage, MeterManager, OptimizerManager, ModelManager
import utils


class TrainRunner(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = GradScaler()
        self.datasets = {}
        self.criterions = {}
        self.optimizer_manager = OptimizerManager(self.scaler)
        self.meter_manager = MeterManager()

        self.set_config()
        self.ema = ExponentialMovingAverage(**self.cfg.ema)
        self.set_datasets()
        self.data_loaders = utils.generate_data_loaders(self.datasets)
        self.models = ModelManager()
        self.set_models()
        self.load_models(self.cfg.train.load_token)
        self.set_optimizers()
        self.models.to_device(self.cfg.model.device)
        self.models.data_parallel()
        self.set_criterions()
        self.set_meters()

    @abstractmethod
    def set_config(self):
        '''
        Set config from config package.

        Example:
            >>> self.cfg = UserDefinedConfig()
        '''
        pass

    @abstractmethod
    def set_datasets(self):
        '''
        Set dataset(s) from dataset package.

        Example:
            >>> self.datasets['train'] = A()
            >>> self.datasets['val'] = B()
        '''
        pass

    @abstractmethod
    def set_models(self):
        '''
        Set model(s) from model package.

        Example:
            >>> self.models['a'] = A()
            >>> self.models['b'] = B()
        '''
        pass

    @abstractmethod
    def set_optimizers(self):
        '''
        Set optimizer(s) for model(s).

        Example:
            >>> self.optimizers['a'] = OptimClass(self.models['a'].parameters(),
                                                  lr=0.1**3)
            >>> self.optimizers['b'] = OptimClass(self.models['b'].parameters(),
                                                  lr=0.1**3)
        '''
        pass

    @abstractmethod
    def set_criterions(self):
        '''
        Set criterion(s) from loss package.
        '''
        pass

    @abstractmethod
    def set_meters(self):
        '''
        Set meter(s) from meter package.
        '''
        pass

    @abstractmethod
    def forward(self, input_data):
        '''
        Forward inference.
        '''
        pass

    @abstractmethod
    def compute_loss(self, input_data, output_data):
        '''
        Compute loss.
        '''
        pass

    @abstractmethod
    def backward(self, loss, is_optimize):
        '''
        Backward propagation and update parameters.
        '''
        pass

    def optimize(self):
        '''
        Update parameters.
        '''
        self.optimizer_manager.step()

    # def apply_ema(self, step):
    #     '''
    #     Apply Exponential Moving Average after optimizing model's weight.
    #     '''
    #     if self.cfg.train.ema == True:
    #         if step == self.cfg.train.ema_start_step:
    #             self.ema.running = True
    #             self.ema.registe(self.models)
    #             start_step_dir = path.join(self.cfg.io.model_dir, str(step))
    #             self.save_models(start_step_dir)

    #         if self.ema.running == True and step % self.cfg.train.ema_update_every == 0:
    #             self.ema.update(self.models)

    def train(self, step, initial_flag):
        '''
        Train phase.
        '''
        if initial_flag == True:
            self.meter_manager.initialize()
            self.models.train()

        self.optimizer_manager.zero_grad()

        for _ in range(self.cfg.train.grad_accumulate_every):
            input_data = next(self.data_loaders['train'])
            input_data['phase'] = 'train'
            input_data = utils.to_device(input_data, self.cfg.model.device)

            if self.cfg.train.amp == True:
                with autocast():
                    output_data = self.forward(input_data)
                    loss = self.compute_loss(input_data, output_data)
            else:
                output_data = self.forward(input_data)
                loss = self.compute_loss(input_data, output_data)

            self.backward(loss)
            self.meter_manager.update(input_data, output_data, loss)

        self.optimize()
        self.ema.step(self.models)
        # self.apply_ema(step)

    @torch.no_grad()
    def eval(self, step):
        '''
        Validate phase.
        '''
        self.meter_manager.initialize()
        self.models.eval()

        with tqdm(self.data_loaders['eval'],
                  desc=f'val-{step}',
                  dynamic_ncols=True,
                  ascii=True,
                  colour='green') as t:
            with self.ema.apply_ema(self.models):
                for input_data in t:
                    input_data['phase'] = 'eval'
                    input_data = utils.to_device(input_data,
                                                 self.cfg.model.device)
                    output_data = self.forward(input_data)
                    loss = self.compute_loss(input_data, output_data)
                    self.meter_manager.update(input_data, output_data, loss)

    def load_models(self, load_token):
        '''
        Load model(s)
        '''
        if self.cfg.train.is_load == True:
            load_dir = path.join(self.cfg.io.model_dir, load_token)

            for name in self.models.keys():
                self.models[name] = utils.load_model(self.models[name],
                                                     load_dir)

            if self.cfg.train.ema == True and self.cfg.train.start_step >= self.cfg.train.ema_start_step:
                load_path = path.join(load_dir, 'ema.pth')
                state_dict = torch.load(load_path, map_location='cpu')
                self.ema.running = True
                self.ema.register(self.models)
                self.ema.load_state_dict(state_dict)

    def save_models(self, save_token):
        '''
        Save model(s).
        '''
        save_dir = path.join(self.cfg.io.model_dir, save_token)

        for value in self.models.values():
            utils.save_model(value, save_dir)

        if self.ema != None:
            ema_dict = self.ema.state_dict()
            save_path = path.join(save_dir, 'ema.pth')
            torch.save(ema_dict, save_path)
