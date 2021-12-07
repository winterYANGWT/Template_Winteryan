from abc import ABC, abstractclassmethod


class TestStep(ABC):
    def __init__(self) -> None:
        super().__init__()
