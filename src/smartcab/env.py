import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np

from ray.rllib.env import MultiAgentEnv

from .enums import SmartCabAction, SmartCabReward, GridSymbol

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

        self.aboard_idx = nb_targets
        self.action_space = Discrete(nb_actions)
        self.observation_space = Tuple(
            [
                Box(0, 14, shape=(2,)),  # veh position (x, y)
                Discrete(nb_targets + 1),  # pass index (+1 in veh)
                Discrete(nb_targets),  # dest index
            ]
        )
        print("Action Space: ", self.action_space)
        print("Observation Space: ", self.observation_space)

        self.max_row = self.height - 1
        self.max_col = self.width - 1
        self.around_coords = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        self.reset()

    @property
    def passenger_loc(self):
        return self.targets[self.state["pass_idx"]]

    @property
    def passenger_dest(self):
        return self.targets[self.state["dest_idx"]]

    @property
    def vehicle_loc(self):
        return (self.state["row"], self.state["col"])

    @property
    def current_cell(self):
        return self.grid[self.state["row"]][self.state["col"]]

    def reset(self):
        self.num_steps = 0
        pass_idx, dest_idx = random.sample(set(range(len(self.targets))), 2)
        self.state = dict(row=11, col=2, pass_idx=pass_idx, dest_idx=dest_idx)
        self.s = self.from_dict(self.state)
        self.last_loc = self.vehicle_loc
        return self.s

    def step(self, action):
        self.num_steps += 1
        state, reward, done = self.actions[action]()
        if self.last_loc != (state["row"], state["col"]):
            self.last_loc = (state["row"], state["col"])
        self.state = state
        self.s = self.from_dict(self.state)
        return self.s, reward, done, {}

    # def to_dict(self, state: list) -> dict:
    #     return dict(
    #         row=state[0][0], col=state[0][1], pass_idx=state[1], dest_idx=state[2]
    #     )

    def from_dict(self, state: dict) -> list:
        return [[state["row"], state["col"]], state["pass_idx"], state["dest_idx"]]

    def default_state(self):
        reward = SmartCabReward.DEFAULT.value
        done = False
        new_state = dict(self.state)
        return new_state, reward, done

    def move_south(self):
        new_state, reward, done = self.default_state()
        if self.current_cell in [GridSymbol.DOWN.value] + GridSymbol.valid_defaults():
            new_row = min(self.state["row"] + 1, self.max_row)
            next_cell = self.grid[new_row][self.state["col"]]
            if (
                next_cell == GridSymbol.BLOCK.value
                or self.last_loc == (new_row, self.state["col"])
                or (
                    self.current_cell == GridSymbol.CROSS.value
                    and next_cell == GridSymbol.RIGHT.value
                )
            ):
                reward = SmartCabReward.ACTION_ERROR.value
            else:
                new_state["row"] = new_row
        else:
            reward = SmartCabReward.ACTION_ERROR.value
        return new_state, reward, done

    def move_north(self):
        new_state, reward, done = self.default_state()
        if self.current_cell in [GridSymbol.UP.value] + GridSymbol.valid_defaults():
            new_row = max(self.state["row"] - 1, 0)
            next_cell = self.grid[new_row][self.state["col"]]
            if (
                next_cell == GridSymbol.BLOCK.value
                or self.last_loc == (new_row, self.state["col"])
                or (
                    self.current_cell == GridSymbol.CROSS.value
                    and next_cell == GridSymbol.DOWN.value
                )
            ):
                reward = SmartCabReward.ACTION_ERROR.value
            else:
                new_state["row"] = new_row
        else:
            reward = SmartCabReward.ACTION_ERROR.value
        return new_state, reward, done

    def move_east(self):
        new_state, reward, done = self.default_state()
        if self.current_cell in [GridSymbol.RIGHT.value] + GridSymbol.valid_defaults():
            new_col = min(self.state["col"] + 1, self.max_col)
            next_cell = self.grid[self.state["row"]][new_col]
            if (
                next_cell == GridSymbol.BLOCK.value
                or self.last_loc == (self.state["row"], new_col)
                or (
                    self.current_cell == GridSymbol.CROSS.value
                    and next_cell == GridSymbol.LEFT.value
                )
            ):
                reward = SmartCabReward.ACTION_ERROR.value
            else:
                new_state["col"] = new_col
        else:
            reward = SmartCabReward.ACTION_ERROR.value
        return new_state, reward, done

    def move_west(self):
        new_state, reward, done = self.default_state()
        if self.current_cell in [GridSymbol.LEFT.value] + GridSymbol.valid_defaults():
            new_col = max(self.state["col"] - 1, 0)
            next_cell = self.grid[self.state["row"]][new_col]
            if (
                next_cell == GridSymbol.BLOCK.value
                or self.last_loc == (self.state["row"], new_col)
                or (
                    self.current_cell == GridSymbol.CROSS.value
                    and next_cell == GridSymbol.RIGHT.value
                )
            ):
                reward = SmartCabReward.ACTION_ERROR.value
            else:
                new_state["col"] = new_col
        else:
            reward = SmartCabReward.ACTION_ERROR.value

        return new_state, reward, done

    def pickup(self):
        new_state, reward, done = self.default_state()
        if self.can_pickup():
            new_state["pass_idx"] = self.aboard_idx
            reward = SmartCabReward.ACTION_OK.value
        else:
            reward = SmartCabReward.ACTION_ERROR.value
        return new_state, reward, done

    def dropoff(self):
        new_state, reward, done = self.default_state()
        if self.can_dropoff():
            new_state["pass_idx"] = self.state["dest_idx"]
            reward = SmartCabReward.ACTION_OK.value
            done = True
        else:
            reward = SmartCabReward.ACTION_ERROR.value
        return new_state, reward, done

    def can_pickup(self):
        return self.state["pass_idx"] < self.aboard_idx and self.around_vehicle(
            self.passenger_loc
        )

    def can_dropoff(self):
        return self.state["pass_idx"] == self.aboard_idx and self.around_vehicle(
            self.passenger_dest
        )

    def around_vehicle(self, location):
        return any(
            [
                location == tuple(np.add(self.vehicle_loc, coords))
                for coords in self.around_coords
            ]
        )


class HierarchicalSmartCabEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.flat_env = SmartCabEnv(env_config)

    def reset(self):
        self.curr_obs = self.flat_env.reset()
        self.curr_goal = None
        self.num_high_level_steps = 0
        self.steps_remaining_at_level = 100
        # Current low level agent unique id.
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        return {
            "high_level_agent": self.curr_obs,
        }

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def _high_level_step(self, action):
        print("High level agent sets goal {}".format(action))
        self.curr_goal = action
        self.curr_rew = 0
        self.num_high_level_steps += 1
        self.low_level_agent_id = "low_level_{}".format(self.num_high_level_steps)
        obs = {self.low_level_agent_id: [self.curr_obs, self.curr_goal]}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _low_level_step(self, action):
        # print("Low level agent step {}".format(action))
        self.steps_remaining_at_level -= 1
        # Map low-level action according to goal
        if action == 4 and self.curr_goal == 1:
            action = 5

        # Environment step
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)
        self.curr_obs = f_obs
        self.curr_rew += f_rew

        # Calculate low-level agent observation and reward
        obs = {self.low_level_agent_id: [f_obs, self.curr_goal]}
        rew = {self.low_level_agent_id: f_rew}

        # Handle env/goal termination and transitions back to higher level
        done = {"__all__": False}
        if f_rew > 0 or self.steps_remaining_at_level == 0:
            print("High level reward {}".format(self.curr_rew))
            rew["high_level_agent"] = self.curr_rew
            obs["high_level_agent"] = f_obs
            if f_done:
                done["__all__"] = True
            else:
                done[self.low_level_agent_id] = True

        return obs, rew, done, {}
