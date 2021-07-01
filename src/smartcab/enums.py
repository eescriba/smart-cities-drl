import numpy as np
from enum import Enum


class SmartCabAction(Enum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5


class SmartCabReward(Enum):
    DEFAULT = -1
    ACTION_OK = 30
    ACTION_ERROR = -10
    MOVE_ERROR = -3


class GridSymbol(Enum):
    DOWN = "▽"
    UP = "△"
    RIGHT = "▷"
    LEFT = "◁"
    CROSS = "o"
    BLOCK = "#"
    TARGET = "X"
    STATION = "+"

    @classmethod
    def valid_defaults(cls):
        return [cls.CROSS.value, cls.TARGET.value, cls.STATION.value]

    @classmethod
    def directions(cls):
        return [cls.DOWN, cls.UP, cls.RIGHT, cls.LEFT, cls.CROSS]

    @property
    def is_direction(self):
        return self in GridSymbol.directions()
