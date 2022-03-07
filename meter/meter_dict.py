from collections import UserDict


class MeterDict(UserDict):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        l = [str(v) for v in self.values()]
        s = ', '.join(l)
        return s

    def __repr__(self) -> str:
        return str(self)
