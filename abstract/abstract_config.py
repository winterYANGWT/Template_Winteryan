from abc import ABC
import os.path as path
import torch

__all__ = ['AbstractConfig']


class AbstractConfig(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.SEED = 0
        self.PREFETCH_FACTOR = 2
        self.NUM_WORKERS = 1
        self.PIN_MEMORY = True
        self.MODEL_DIR = path.join(path.dirname(__file__), '../Data/Models')
        self.DATA_DIR = path.join(path.dirname(__file__), '../Data/Datasets')
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.DATA_PARALLEL = torch.cuda.device_count() > 1
        torch.set_printoptions(precision=8)
