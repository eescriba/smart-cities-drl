import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from .enums import TaxiAction, TaxiReward


class MobilityEnv:
    """ Mixin """

    def __init__(self, file_path):
        with open(file_path, "r") as f:
            self.grid = [[e for e in line[:-1].split(" ")] for line in f]

        f.close()
        self.targets = []
        self.cells = []
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                coords = (i, j)
                if cell == "X":
                    self.targets.append(coords)
                elif cell != "#":
                    self.cells.append(coords)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])

    @property
    def shape(self):
        return self.height, self.width


class TaxiNetEnv(MobilityEnv, DiscreteEnv):
    """
    Actions
    6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pick up passenger
    - 5: drop off passenger

    Rewards
    4 possible rewards:
    - Default per-step reward: -1
    - Deliver passenger: +30
    - Wrong pickup or drop off: -20
    - Illegal move: -50

    Observations
    16200 discrete states:
    - 225 vehicle locations
    - 8 target locations
    - 9 passenger locations

    State space
    One vehicle - One passenger:
        (vehicle_x, vehicle_y, passenger_loc, target)
    """

    # State = namedtuple("State", "row col pass_idx dest_idx reward done")

    def __init__(self, file_path="resources/city5.txt"):
        MobilityEnv.__init__(self, file_path)

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
        self.max_row = self.height - 1
        self.max_col = self.width - 1
        self.dims = (self.height, self.width, nb_targets + 1, nb_targets)

        P, initial_state_distrib = self.init_discrete_env(
            nb_states, nb_actions, nb_targets
        )

        DiscreteEnv.__init__(self, nb_states, nb_actions, P, initial_state_distrib)

    def init_discrete_env(self, nb_states, nb_actions, nb_targets):
        P = {
            state: {action: [] for action in range(nb_actions)}
            for state in range(nb_states)
        }
        initial_state_distrib = np.zeros(nb_states)
        for row in range(self.height):
            for col in range(self.width):
                for pass_idx in range(nb_targets + 1):
                    for dest_idx in range(nb_targets):
                        state_dict = dict(
                            row=row, col=col, pass_idx=pass_idx, dest_idx=dest_idx
                        )
                        state = self.encode(list(state_dict.values()))
                        if pass_idx < nb_targets and pass_idx != dest_idx:
                            initial_state_distrib[state] += 1
                        for action in range(nb_actions):
                            new_state, reward, done = self.next_state(
                                action, state_dict
                            )
                            P[state][action].append((1.0, new_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        return P, initial_state_distrib

    def encode(self, state: list) -> TaxiAction:
        return np.ravel_multi_index(state, self.dims)

    def decode(self, i: int) -> list:
        return np.unravel_index(i, self.dims)

    def next_state(self, action: int, state: dict):
        state, reward, done = self.actions[action](state)
        return self.encode(list(state.values())), reward, done

    def default_state(self, state):
        reward = TaxiReward.DEFAULT.value
        done = False
        new_state = dict(state)
        return new_state, reward, done

    def move_south(self, state):
        new_state, reward, done = self.default_state(state)
        new_state["row"] = min(state["row"] + 1, self.max_row)
        return new_state, reward, done

    def move_north(self, state):
        new_state, reward, done = self.default_state(state)
        new_state["row"] = max(state["row"] - 1, 0)
        return new_state, reward, done

    def move_east(self, state):
        new_state, reward, done = self.default_state(state)
        new_state["col"] = min(state["col"] + 1, self.max_col)
        return state, reward, done

    def move_west(self, state):
        new_state, reward, done = self.default_state(state)
        new_state["col"] = max(state["col"] - 1, 0)
        return new_state, reward, done

    def pickup(self, state):
        new_state, reward, done = self.default_state(state)
        vehicle_loc = (state["row"], state["col"])
        if (
            state["pass_idx"] < self.aboard_idx
            and vehicle_loc == self.targets[state["pass_idx"]]
        ):
            new_state["pass_idx"] = self.aboard_idx
        else:  # passenger not at location
            reward = TaxiReward.ACTION_ERROR.value
        return new_state, reward, done

    def dropoff(self, state):
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