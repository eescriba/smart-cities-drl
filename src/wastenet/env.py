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
        Current day: D - 1

    """

    def __init__(self, env_config):

        self.G = env_config.get("graph", generate_graph())
        self.nb_nodes = self.G.number_of_nodes()
        self.nb_dumpsters = self.nb_nodes - 2
        self.start_node = 0
        self.end_node = self.nb_nodes - 1
        self.fill_ranges = env_config.get("fill_ranges", generate_fill_ranges())
        self.total_days = env_config.get("nb_days", 30)

        # Gym
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(self.nb_nodes),
                spaces.Box(
                    np.array([0 for _ in range(self.nb_dumpsters)]),
                    np.array([100 for _ in range(self.nb_dumpsters)]),
                    dtype=np.uint8,
                ),
            ]
        )
        self.s = self.reset()

    def reset(self):
        self.total_reward = 0
        self.total_dist = 0
        self.total_collected = 0
        self.total_overflow = 0
        self.mean_reward = 0
        self.mean_dist = 0
        self.mean_collected = 0
        self.mean_overflow = 0
        self.current_node = self.start_node
        self.current_day = 0
        self.current_path = [self.start_node]
        self.fill_levels = [random.randrange(*fr) for fr in self.fill_ranges]
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
            self._update_mean_stats()
        elif self.current_node == self.end_node:
            dist = self._update_path()
            reward += Reward.ROUTE_FINISH
            reward += Reward.MOVE * dist
            self.total_dist += dist
        else:
            dumpster_idx = self.current_node - 1
            fill = random.randrange(*self.fill_ranges[dumpster_idx])
            if action == Action.PICKUP:
                dist = self._update_path()
                reward += Reward.PICKUP
                reward += Reward.MOVE * dist
                self.fill_levels[dumpster_idx] = fill
                self.total_dist += dist
                self.total_collected += 1
            else:
                self.fill_levels[dumpster_idx] = min(
                    100, self.fill_levels[dumpster_idx] + fill
                )

            if self.fill_levels[dumpster_idx] == 100:
                reward += Reward.OVERFLOW
                self.total_overflow += 1

        self.s = [self.current_node, self.fill_levels]
        self.total_reward += reward
        return self.s, reward, done, {}

    def _update_mean_stats(self):
        self.mean_reward = self.total_reward / self.current_day
        self.mean_dist = self.total_dist / self.current_day
        self.mean_overflow = self.total_overflow / self.current_day
        self.mean_collected = self.total_collected / self.current_day

    def _update_path(self):
        dist, path = single_source_dijkstra(
            self.G, source=self.current_path[-1], target=self.current_node
        )
        self.current_path += path[1:]
        return dist
