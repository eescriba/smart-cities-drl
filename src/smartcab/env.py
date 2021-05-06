import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np

from ray.rllib.env import MultiAgentEnv

from .enums import TaxiAction, TaxiReward

logger = logging.getLogger(__name__)


# MAP_DATA = """
# X o o o X
# o o o o o
# o o o o o
# o o o o o
# X o o X o"""

MAP_DATA = """
# # # # # # # # # # # # # # #
# # # # ▽ ◁ ◁ ◁ ◁ ◁ ◁ ◁ ◁ ◁ ◁
# # # X ▽ # # # # # ▽ # X # △
▷ ▷ ▷ ▷ o ▷ ▷ ▷ ▷ ▷ o ▷ ▷ ▷ o
△ # # # ▽ # # X # # ▽ △ # # ▽
△ # # # ▽ # # # # # ▽ △ # # ▽
△ X # # ▽ # # # # # ▽ △ # # ▽
△ # # # ▽ # # # # # ▽ △ # # ▽
△ # # # ▽ # # # # # ▽ △ # # ▽
△ ◁ ◁ ◁ o ◁ ◁ ◁ ◁ ◁ o o ◁ ◁ o
△ + + + ▽ # # # # # o o ▷ ▷ o
△ + + + ▽ # # # # # ▽ △ # X ▽
△ + + + ▽ # # # # # ▽ △ # # ▽
△ # # # ▽ # # # # X ▽ △ # # ▽
△ ◁ ◁ ◁ ◁ ◁ ◁ ◁ ◁ ◁ o o ◁ ◁ o"""


class SmartCabEnv(gym.Env):
    def __init__(self, env_config):
        self.grid = [
            [e for e in line.split(" ")] for line in MAP_DATA.split("\n") if line
        ]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        logger.info("Loaded map {} {}".format(self.height, self.width))
        self.targets = []
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                coords = (i, j)
                if cell == "X":
                    self.targets.append(coords)

        print(self.targets)
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
        # nb_states = self.height * self.width * nb_targets * (nb_targets + 1)

        self.aboard_idx = nb_targets
        # self.pass_idx, self.dest_idx = 0, 0
        self.action_space = Discrete(nb_actions)
        # self.observation_space = Discrete(nb_states)
        self.observation_space = Tuple(
            [
                Box(0, 14, shape=(2,)),  # veh position (x, y)
                Discrete(nb_targets),  # pass index
                Discrete(nb_targets - 1),  # dest index
            ]
        )
        print("Action Space: ", self.action_space)
        print("Observation Space: ", self.observation_space)

        self.max_row = self.height - 1
        self.max_col = self.width - 1

        self.reset()
        # self.dims = (5, 5, 5, 4)
        # self.state = dict(row=3, col=2, pass_idx=0, dest_idx=3)
        # self.s = self.encode(self.state)

    @property
    def passenger_loc(self):
        return self.targets[self.state["pass_idx"]]

    @property
    def passenger_dest(self):
        return self.targets[self.state["dest_idx"]]

    @property
    def vehicle_loc(self):
        return (self.state["row"], self.state["col"])

    def reset(self):
        self.num_steps = 0
        # self.pos = (11, 2)
        pass_idx, dest_idx = random.sample(set(range(len(self.targets))), 2)
        self.state = dict(row=11, col=2, pass_idx=pass_idx, dest_idx=dest_idx)
        self.s = self.from_dict(self.state)
        return self.s
        # return [[self.pos[0], self.pos[1]], self.pass_idx, self.dest_idx]

    def step(self, action):
        self.num_steps += 1
        state, reward, done = self.actions[action](self.state)
        # done = done or self.num_steps > 200
        self.state = state
        # self.s = self.encode(state)
        self.s = self.from_dict(self.state)
        return self.s, reward, done, {}

    # def encode(self, state: dict) -> TaxiAction:
    #     return np.ravel_multi_index(list(state.values()), self.dims)

    # def decode(self, s: int) -> list:
    #     return np.unravel_index(s, self.dims)

    def to_dict(self, state: list) -> dict:
        return dict(
            row=state[0][0], col=state[0][1], pass_idx=state[1], dest_idx=state[2]
        )

    def from_dict(self, state: dict) -> list:
        return [[state["row"], state["col"]], state["pass_idx"], state["dest_idx"]]

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


# class HierarchicalSmartCabEnv(MultiAgentEnv):
#     def __init__(self, env_config):
#         self.flat_env = SmartCabEnv(env_config)

#     def reset(self):
#         self.cur_obs = self.flat_env.reset()
#         self.current_goal = None
#         self.steps_remaining_at_level = None
#         self.num_high_level_steps = 0
#         # current low level agent id. This must be unique for each high level
#         # step since agent ids cannot be reused.
#         self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
#         return {
#             "high_level_agent": self.cur_obs,
#         }

#     def step(self, action_dict):
#         assert len(action_dict) == 1, action_dict
#         if "high_level_agent" in action_dict:
#             return self._high_level_step(action_dict["high_level_agent"])
#         else:
#             return self._low_level_step(list(action_dict.values())[0])

#     def _high_level_step(self, action):
#         logger.debug("High level agent sets goal {}".format(action))
#         self.current_goal = action
#         self.steps_remaining_at_level = 25
#         self.num_high_level_steps += 1
#         self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
#         obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
#         rew = {self.low_level_agent_id: 0}
#         done = {"__all__": False}
#         return obs, rew, done, {}

#     def _low_level_step(self, action):
#         logger.debug("Low level agent step {}".format(action))
#         self.steps_remaining_at_level -= 1
#         cur_pos = (self.flat_env.state["row"], self.flat_env.state["col"])
#         f_obs, f_rew, f_done, _ = self.flat_env.step(action)
#         new_pos = (self.flat_env.state["row"], self.flat_env.state["col"])
#         self.cur_obs = f_obs

#         # Calculate low-level agent observation and reward
#         obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
#         rew = {self.low_level_agent_id: f_rew}

#         # Handle env termination & transitions back to higher level
#         done = {"__all__": False}
#         if f_done:
#             done["__all__"] = True
#             logger.debug("high level final reward {}".format(f_rew))
#             rew["high_level_agent"] = f_rew
#             obs["high_level_agent"] = f_obs
#         elif self.steps_remaining_at_level == 0:
#             done[self.low_level_agent_id] = True
#             rew["high_level_agent"] = 0
#             obs["high_level_agent"] = f_obs

#         return obs, rew, done, {}
