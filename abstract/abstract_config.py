from abc import ABC
import os.path as path

__all__ = ['AbstractConfig']


class AbstractConfig(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.PREFETCH_FACTOR = 2
        self.NUM_WORKERS = 1
        self.MODEL_DIR = path.join(path.dirname(__file__), '../Data/Models')
        self.DATA_DIR = path.join(path.dirname(__file__), '../Data/Datasets')
