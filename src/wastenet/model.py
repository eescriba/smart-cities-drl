import math
import networkx as nx
from enum import Enum

from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from .agents import DumpsterAgent, BaseAgent
from .env import WasteNetEnv
from .scheduler import WasteNetActivation

# from .rl_agents import ppo


def nb_empty(model):
    return sum(map(lambda l: l <= 0.2, model.env.fill_levels))


def nb_medium(model):
    return sum(map(lambda l: l > 0.2 and l < 0.8, model.env.fill_levels))


def nb_full(model):
    return sum(map(lambda l: l >= 0.8 and l < 1.0, model.env.fill_levels))


def nb_overflow(model):
    return sum(map(lambda l: l == 1.0, model.env.fill_levels))


class WasteNet(Model):
    """Waste collection network model"""

    def __init__(self, nb_nodes=7):
        self.nb_nodes = nb_nodes

        # Network
        self.G = nx.path_graph(nb_nodes)
        G = nx.Graph()

        G.add_edge(0, 1, weight=2)
        G.add_edge(0, 2, weight=2)
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=2)
        G.add_edge(1, 4, weight=2)
        G.add_edge(2, 3, weight=2)
        G.add_edge(2, 5, weight=2)
        G.add_edge(3, 4, weight=2)
        G.add_edge(3, 5, weight=2)
        G.add_edge(4, 5, weight=2)
        G.add_edge(4, 6, weight=2)
        G.add_edge(5, 6, weight=2)

        self.G = G
        self.grid = NetworkGrid(self.G)

        # Gym Environment
        env_config = {}
        self.env = WasteNetEnv(env_config)

        # RL Agent
        ray.init(ignore_reinit_error=True)
        ppo = PPOTrainer(DEFAULT_CONFIG.copy(), env=WasteNetEnv)
        ppo.restore("./checkpoints/checkpoint-143")

        # Scheduler
        self.schedule = WasteNetActivation(self, rl_agent=ppo)

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
            if i == 0:
                a = BaseAgent(0, self)
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

    # def run_model(self, n):
    #     for i in range(n):
    #         self.step()
