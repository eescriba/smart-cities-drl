import numpy as np
from enum import IntEnum, Enum


class WasteNetAction(IntEnum):
    AVOID = 0
    PICKUP = 1


class WasteNetReward(IntEnum):
    MOVE = -1
    PICKUP = -3
    OVERFLOW = -25
    ROUTE_FINISH = 30


class WasteNetMode(Enum):
    RANDOM = 0
    COMPLETE = 1
    PARTIAL = 2
    PPO = 3

    @classmethod
    def names(cls):
        return [mode.name for mode in cls]
