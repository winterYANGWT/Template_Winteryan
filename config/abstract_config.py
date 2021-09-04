from abc import ABC, abstractclassmethod


__all__=['AbstractConfig']


class AbstractConfig(ABC):
    def __init__(self) -> None:
        super().__init__()
