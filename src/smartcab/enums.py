import numpy as np
from enum import Enum


class TaxiAction(Enum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5


class TaxiReward(Enum):
    DEFAULT = -1
    ACTION_OK = 30
    ACTION_ERROR = -20
    ILLEGAL_MOVE = -50


class GridSymbol(Enum):
    DOWN = "▽"
    UP = "△"
    RIGHT = "▷"
    LEFT = "◁"
    CROSS = "o"
    BLOCK = "#"
    TARGET = "X"
    STATION = "+"

    @property
    def directions(self):
        return [self.DOWN, self.UP, self.RIGHT, self.LEFT, self.CROSS]

    @property
    def is_direction(self):
        return self in self.directions
