import random
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
from ray.rllib.env import MultiAgentEnv

from .enums import SmartCabReward, GridSymbol


MAP_DATA = """
▷ ▷ ▷ ▷ ▽ # # #
△ X # # ▽ # # #
△ # # X ▽ # X #
△ ◁ ◁ ◁ o ▷ ▷ ▽
o ▷ ▷ ▷ ▽ # # ▽
△ + # # ▽ # # ▽
△ # # # ▽ X # ▽
△ ◁ ◁ ◁ ◁ ◁ ◁ ◁"""


class SmartCabEnv(gym.Env):
    def __init__(self, env_config):
        self.grid = [
            [e for e in line.split(" ")] for line in MAP_DATA.split("\n") if line
        ]
        self.height = len(self.grid)
        self.width = len(self.grid[0])
        self.max_row = self.height - 1
        self.max_col = self.width - 1
        self.base = (self.max_row, self.max_col)
        self.targets = []
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                coords = (i, j)
                if cell == "X":
                    self.targets.append(coords)
                elif cell == "+":
                    self.base = coords

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
                Box(0, 7, shape=(2,)),  # veh position (x, y)
                Discrete(nb_targets + 1),  # pass index (+1 in veh)
                Discrete(nb_targets),  # dest index
            ]
        )

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
        pass_idx, dest_idx = self.new_passenger()
        self.state = dict(
            row=self.base[0],
            col=self.base[1],
            pass_idx=pass_idx,
            dest_idx=dest_idx,
        )
        self.s = self.from_dict(self.state)
        self.last_loc = self.vehicle_loc
        return self.s

    def step(self, action):
        self.num_steps += 1
        state, reward, done = self.actions[action]()
        if (state["row"], state["col"]) not in [self.last_loc, self.vehicle_loc]:
            self.last_loc = (self.state["row"], self.state["col"])
        self.state = state
        self.s = self.from_dict(self.state)
        return self.s, reward, done, {}

    def from_dict(self, state: dict) -> list:
        return [
            [state["row"], state["col"]],
            state["pass_idx"],
            state["dest_idx"],
        ]

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
                reward = SmartCabReward.MOVE_ERROR.value
            else:
                new_state["row"] = new_row
        else:
            reward = SmartCabReward.MOVE_ERROR.value
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
                reward = SmartCabReward.MOVE_ERROR.value
            else:
                new_state["row"] = new_row
        else:
            reward = SmartCabReward.MOVE_ERROR.value
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
                reward = SmartCabReward.MOVE_ERROR.value
            else:
                new_state["col"] = new_col
        else:
            reward = SmartCabReward.MOVE_ERROR.value
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
                reward = SmartCabReward.MOVE_ERROR.value
            else:
                new_state["col"] = new_col
        else:
            reward = SmartCabReward.MOVE_ERROR.value

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
            [self.passenger_loc]
        )

    def can_dropoff(self):
        return self.state["pass_idx"] == self.aboard_idx and self.around_vehicle(
            [self.passenger_dest]
        )

    def around_vehicle(self, locations):
        return any(
            [
                tuple(np.add(self.vehicle_loc, coords)) in locations
                for coords in self.around_coords
            ]
        )

    def new_passenger(self):
        return random.sample(set(range(len(self.targets))), 2)


class HierarchicalSmartCabEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.flat_env = SmartCabEnv(env_config)
        self.goal_action_space = Discrete(2)
        self.action_action_space = Discrete(2)
        self.move_action_space = Discrete(4)
        self.reset()

    def reset(self):
        self.curr_obs = self.flat_env.reset()
        self.curr_goal = None
        self.num_goal_level_steps = 0
        self.num_action_level_steps = 0
        self.steps_remaining = 50
        # Current low levels agents with unique id.
        self.action_level_agent_id = "action_level_{}".format(self.num_goal_level_steps)
        self.move_level_agent_id = "move_level_{}".format(self.num_action_level_steps)
        return {
            "goal_level_agent": self.curr_obs,
        }

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "goal_level_agent" in action_dict:
            return self._goal_level_step(action_dict["goal_level_agent"])
        elif "action_level_agent" in action_dict:
            return self._action_level_step(action_dict["action_level_agent"])
        else:
            return self._move_level_step(list(action_dict.values())[0])

    def _goal_level_step(self, action):
        self.curr_goal = action
        self.curr_rew = 0
        self.num_goal_level_steps += 1
        self.action_level_agent_id = "action_level_{}".format(self.num_goal_level_steps)
        obs = {self.action_level_agent_id: [self.curr_obs, self.curr_goal]}
        rew = {self.action_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _action_level_step(self, action):
        self.num_action_level_steps += 1
        # Move action
        if action == 0:
            self.move_level_agent_id = "move_level_{}".format(
                self.num_action_level_steps
            )
            obs = {self.move_level_agent_id: [self.curr_obs, self.curr_goal]}
            rew = {self.move_level_agent_id: 0}
            done = {"__all__": False}
        # Pick or drop action
        else:
            # Map low-level action
            action += 4
            # Environment step
            self.steps_remaining -= 1
            f_obs, f_rew, f_done, _ = self.flat_env.step(action)
            self.curr_obs = f_obs
            self.curr_rew += f_rew
            # Calculate action-level agent observation and reward
            obs = {self.action_level_agent_id: [f_obs, self.curr_goal]}
            rew = {self.action_level_agent_id: f_rew}
            # Handle env/goal termination and transitions back to higher level
            if f_rew > 0 or self.steps_remaining == 0:
                rew["goal_level_agent"] = self.curr_rew
                obs["goal_level_agent"] = f_obs
                if f_done:
                    done["__all__"] = True
                else:
                    done[self.action_level_agent_id] = True
        return obs, rew, done, {}

    def _move_level_step(self, action):
        # Environment step
        self.steps_remaining -= 1
        f_obs, f_rew, f_done, _ = self.flat_env.step(action)
        self.curr_obs = f_obs
        self.curr_rew += f_rew
        # Calculate move-level agent observation and reward
        obs = {self.move_level_agent_id: [f_obs, self.curr_goal]}
        rew = {self.move_level_agent_id: f_rew}
        # Handle env/goal termination and transitions back to higher level
        done = {"__all__": False}
        if f_done or self.steps_remaining == 0:
            done[self.move_level_agent_id] = True
        return obs, rew, done, {}
