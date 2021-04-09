import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np

from ray.rllib.env import MultiAgentEnv

from .enums import TaxiAction, TaxiReward

logger = logging.getLogger(__name__)

# Agent has to traverse the maze from the starting position S -> F
# Observation space [x_pos, y_pos, wind_direction]
# Action space: stay still OR move in current wind direction
MAP_DATA = """
X o o o X
o o o o o
o o o o o
o o o o o
X o o X o"""

# (vehicle_x, vehicle_y, passenger_loc, target)

class SmartCabEnv(gym.Env):
    def __init__(self, env_config):
        self.grid = [[e for e in line.split(" ")] for line in MAP_DATA.split("\n") if line]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        logger.info("Loaded map {} {}".format(self.height, self.width))
        self.targets = []
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                coords = (i, j)
                if cell == "X":
                    self.targets.append(coords)

        self.actions = {
            0: self.move_south,
            1: self.move_north,
            2: self.move_east,
            3: self.move_west,
            4: self.pickup,
            5: self.dropoff,
        }
        nb_actions = len(self.actions)
        nb_targets = len(self.targets)
        nb_states = self.height * self.width * nb_targets * (nb_targets + 1)

        print("Actions: ", nb_actions)
        print("Targets: ", nb_targets)
        print("States: ", nb_states)

        self.aboard_idx = nb_targets
        self.action_space = Discrete(nb_actions) 
        self.observation_space = Discrete(nb_states)
        self.dims = (5, 5, 5, 4)
        self.state = dict(row=2, col=2, pass_idx=0, dest_idx=3)
        self.s = self.encode(self.state)
        self.max_row = self.height - 1
        self.max_col = self.width - 1

    def reset(self):
        self.pass_idx = 0
        self.dest_idx = 3
        self.row = 2
        self.col = 2
        self.num_steps = 0
        self.state = dict(row=2, col=2, pass_idx=0, dest_idx=3)
        self.s = self.encode(self.state)
        return self.s

    def step(self, action):
        state, reward, done = self.actions[action](self.state)
        self.state = state
        self.s = self.encode(state)
        return self.s, reward, done, {}

    def encode(self, state: dict) -> TaxiAction:
        return np.ravel_multi_index(list(state.values()), self.dims)
    
    def decode(self, s: int) -> list:
        return np.unravel_index(s, self.dims)

    def default_state(self, state: dict):
        reward = TaxiReward.DEFAULT.value
        done = False
        new_state = dict(state)
        return new_state, reward, done

    def move_south(self, state: dict):
        new_state, reward, done = self.default_state(state)
        new_state["row"] = min(state["row"] + 1, self.max_row)
        return new_state, reward, done

    def move_north(self, state: dict):
        new_state, reward, done = self.default_state(state)
        new_state["row"] = max(state["row"] - 1, 0)
        return new_state, reward, done

    def move_east(self, state: dict):
        new_state, reward, done = self.default_state(state)
        new_state["col"] = min(state["col"] + 1, self.max_col)
        return state, reward, done

    def move_west(self, state: dict):
        new_state, reward, done = self.default_state(state)
        new_state["col"] = max(state["col"] - 1, 0)
        return new_state, reward, done

    def pickup(self, state: dict):
        new_state, reward, done = self.default_state(state)
        vehicle_loc = (state["row"], state["col"])
        if (
            state["pass_idx"] < self.aboard_idx
            and vehicle_loc == self.targets[state["pass_idx"]]
        ):
            state["pass_idx"] = self.aboard_idx
        else:  # passenger not at location
            reward = TaxiReward.ACTION_ERROR.value
        return new_state, reward, done

    def dropoff(self, state: dict):
        new_state, reward, done = self.default_state(state)
        vehicle_loc = (state["row"], state["col"])
        if (vehicle_loc == self.targets[state["dest_idx"]]) and state[
            "pass_idx"
        ] == self.aboard_idx:
            new_state["pass_idx"] = state["dest_idx"]
            done = True
            reward = TaxiReward.ACTION_OK.value
        elif (vehicle_loc in self.targets) and state["pass_idx"] == self.aboard_idx:
            new_state["pass_idx"] = self.targets.index(vehicle_loc)
        else:  # dropoff at wrong location
            reward = TaxiReward.ACTION_ERROR.value
        return new_state, reward, done
