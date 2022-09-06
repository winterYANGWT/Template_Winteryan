import os.path as path
from ml_collections import ConfigDict
import torch

__all__ = ['AbstractConfig']


class AbstractConfig(ConfigDict):
    def __init__(self) -> None:
        super().__init__()
        self.train = ConfigDict()
        self.eval = ConfigDict()
        self.test = ConfigDict()
        self.data = ConfigDict()
        self.model = ConfigDict()
        self.io = ConfigDict()
        self.ema = ConfigDict()

        # io
        self.io.model_dir = path.join(path.dirname(__file__), '../Data/Models')
        self.io.data_dir = path.join(path.dirname(__file__),
                                     '../Data/Datasets')
        self.io.result_dir = path.join(path.dirname(__file__),
                                       '../Data/Results')

        # data
        self.data.prefetch_factor = 2
        self.data.is_pin_memory = True
        self.data.num_workers = 1

        # model
        self.model.is_data_parallel = torch.cuda.device_count() > 1
        self.model.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # train
        self.train.seed = 0
        self.train.amp = True
        self.train.grad_accumulate_every = 1

        # ema
        self.ema.use_ema = False
        self.ema.start_step = 2000
        self.ema.update_every = 1
        self.ema.beta = 0.99

        torch.set_printoptions(precision=8)
