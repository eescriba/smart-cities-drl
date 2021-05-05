import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class WasteNetEnv(gym.Env):
    """
    Description:

    Source:

    Observation:
        Type: Tuple(Discrete(N), Box(N))
        Num     Observation     Min     Max
        0-(N+1) Current node
        0-N     Fill level      0.0     1.0

    Actions:
        Type: Discrete(3)
        Num     Action
        0       Avoid next dumpster
        1       Pickup next dumpster

    Rewards:
        Type: int
        Reward      Value
        Move        -1 * Distance
        Pickup:     -2 * Dumpster
        Overflow:   -20
        Finish:     +20

    Starting State:
        Fill level: random
        Current node: 0
        Current day: 0

    Episode Termination:
        Current node: N
        Current day: 6

    """

    def __init__(self, env_config):

        self.nb_nodes = 7
        self.nb_dumpsters = self.nb_nodes - 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(self.nb_nodes),
                spaces.Box(
                    np.array([0.0 for _ in range(self.nb_dumpsters)]),
                    np.array([1.0 for _ in range(self.nb_dumpsters)]),
                    dtype=np.float32,
                ),
            ]
        )
        self.seed()
        self.s = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_node = 0
        self.current_day = 0
        self.fill_levels = [0.5, 0.2, 0.3, 0.1, 0.6, 0.2]
        return [self.current_node, self.fill_levels]

    def step(self, action):
        reward = -3
        done = False

        self.current_node = (self.current_node + 1) % self.nb_nodes

        if self.current_node == 0:  # base
            reward += 25
            self.current_day = (self.current_day + 1) % 7
            if self.current_day == 6:
                done = True
        else:
            dumpster_idx = self.current_node - 1
            if action == 0:
                self.fill_levels[dumpster_idx] = min(
                    1.0, self.fill_levels[dumpster_idx] + 0.3
                )
            else:
                self.fill_levels[dumpster_idx] = 0.3
                reward -= 2

            if self.fill_levels[dumpster_idx] == 1.0:
                reward -= 20

        self.s = [self.current_node, self.fill_levels]
        return self.s, reward, done, {}
