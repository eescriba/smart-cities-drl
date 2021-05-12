import random
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra

from .enums import WasteNetAction as Action, WasteNetReward as Reward
from .utils import generate_graph, generate_fill_ranges


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
        Pickup:     -2
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

        self.G = env_config.get("graph", generate_graph())
        self.nb_nodes = self.G.number_of_nodes()
        self.nb_dumpsters = self.nb_nodes - 2
        self.start_node = 0
        self.end_node = self.nb_nodes - 1
        self.fill_ranges = env_config.get("fill_ranges", generate_fill_ranges())
        self.total_days = env_config.get("nb_days", 15)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(self.nb_nodes),
                spaces.Box(
                    np.array([np.float32(0.0) for _ in range(self.nb_dumpsters)]),
                    np.array([np.float32(1.0) for _ in range(self.nb_dumpsters)]),
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
        self.current_node = self.start_node
        self.current_day = 0
        self.current_path = [self.start_node]
        self.fill_levels = [random.randrange(*fr) / 10 for fr in self.fill_ranges]
        return [self.current_node, self.fill_levels]

    def step(self, action):
        reward = 0
        done = False
        self.current_node = (self.current_node + 1) % self.nb_nodes

        if self.current_node == self.start_node:
            self.current_path = [self.start_node]
            self.current_day += 1
            if self.current_day == self.total_days:
                done = True
        elif self.current_node == self.end_node:
            dist = self._update_path()
            reward += Reward.ROUTE_FINISH
            reward += Reward.MOVE * dist
        else:
            dumpster_idx = self.current_node - 1
            fill = random.randrange(*self.fill_ranges[dumpster_idx]) / 10
            if action == Action.PICKUP:
                dist = self._update_path()
                reward += Reward.PICKUP
                reward += Reward.MOVE * dist
                self.fill_levels[dumpster_idx] = fill
            else:
                self.fill_levels[dumpster_idx] = min(
                    1.0, self.fill_levels[dumpster_idx] + fill
                )

            if self.fill_levels[dumpster_idx] == 1.0:
                reward += Reward.OVERFLOW

        self.s = [self.current_node, self.fill_levels]
        return self.s, reward, done, {}

    def _update_path(self):
        dist, path = single_source_dijkstra(
            self.G, source=self.current_path[-1], target=self.current_node
        )
        self.current_path += path[1:]
        return dist
