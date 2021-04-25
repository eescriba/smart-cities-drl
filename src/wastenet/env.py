import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class WasteNetEnv(gym.Env):
    """
    Description:

    Source:

    Observation:
        Type: Tuple(Box, Discrete)
            Type: Box(N)
            Num     Observation     Min     Max
            0-N     Fill level      0.0     1.0
            Type: Discrete(N)
            Num     Observation
            0-N     Current node

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
        Current day: 7

    """

    def __init__(self):

        self.nb_dumpsters = 6
        self.nb_nodes = self.nb_dumpsters + 1
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
        self.current_node = 0
        self.current_day= 0
        self.fill_levels = [0.5, 0.2, 0.3, 0.1, 0.6, 0.2]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = -3
        done = False
        self.current_node = (self.current_node + 1) % self.nb_nodes
        self.current_day = (self.current_day + 1) % 7

        if action == 0:
            self.fill_levels[self.N] += 0.4
        else:
            self.fill_levels[self.N] = 0.4
            reward -= 2

        if self.current_node == self.nb_dumpsters:
            reward+=20
            if self.current_day == 6:
                done = True
        
        if self.fill_levels[self.current_node] > 1.0:
            reward-=20

        return [self.fill_levels, self.current_node], reward, done, {}

    def reset(self):
        self.current_node = 0
        self.fill_levels = [0.5, 0.2, 0.3, 0.1, 0.6, 0.2]

