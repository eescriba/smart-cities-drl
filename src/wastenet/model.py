import math
from enum import Enum

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from .agents import DumpsterAgent, BaseAgent
from .env import WasteNetEnv
from .utils import generate_graph
from .scheduler import WasteNetActivation


class WasteNet(Model):
    """Waste collection network model"""

    def __init__(self, mode, nb_nodes=10, nb_days=15):

        # Network
        self.G = generate_graph(nb_nodes)
        self.grid = NetworkGrid(self.G)

        # Gym Environment
        env_config = dict(graph=self.G, nb_days=nb_days)
        self.env = WasteNetEnv(env_config)

        # RL Agent
        ray.init(ignore_reinit_error=True)
        rl_agent = PPOTrainer(DEFAULT_CONFIG.copy(), env=WasteNetEnv)
        # ppo.restore("./checkpoints/checkpoint-143")

        # Scheduler
        self.schedule = WasteNetActivation(self, mode, rl_agent)

        # Data Collector
        self.datacollector = DataCollector(
            {
                "Empty": nb_empty,
                "Medium": nb_medium,
                "Full": nb_full,
                "Overflow": nb_overflow,
            }
        )

        # Mesa Agents
        for i, node in enumerate(self.G.nodes()):
            if i in (0, self.G.number_of_nodes() - 1):
                a = BaseAgent(i, self)
            else:
                a = DumpsterAgent(i, self, self.env.fill_levels[i - 1])
            self.schedule.add(a)
            self.grid.place_agent(a, node)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        done = self.schedule.step()
        self.datacollector.collect(self)
        if done:
            self.running = False


def nb_empty(model):
    return sum(map(lambda l: l <= 0.2, model.env.fill_levels))


def nb_medium(model):
    return sum(map(lambda l: l > 0.2 and l < 0.8, model.env.fill_levels))


def nb_full(model):
    return sum(map(lambda l: l >= 0.8 and l < 1.0, model.env.fill_levels))


def nb_overflow(model):
    return sum(map(lambda l: l == 1.0, model.env.fill_levels))
