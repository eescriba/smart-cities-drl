import random
from collections import namedtuple
from itertools import repeat

from mesa import Model
from mesa.datacollection import DataCollector

import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

from .agents import GridAgent, PassengerAgent, TargetAgent, VehicleAgent
from .env import SmartCabEnv
from .scheduler import SmartCabActivation
from .space import MobilityMultiGrid


class MobilityModel(Model):
    def __init__(
        self,
        rl_agent,
        nb_vehicles,
        nb_targets,
        nb_passengers,
        show_symbols,
        width=15,
        height=15,
    ):

        super().__init__()

        self.schedule = SmartCabActivation(self, rl_agent)
        self.grid = MobilityMultiGrid(width, height, True)
        self.show_symbols = show_symbols

        env_config = {}
        self.env = SmartCabEnv(env_config)

        ray.init(ignore_reinit_error=True)
        dqn = DQNTrainer(DEFAULT_CONFIG.copy(), env=SmartCabEnv)

        self.schedule = SmartCabActivation(self, rl_agent=dqn)

        self._init_environment(self.env)

        self.datacollector = DataCollector(
            model_reporters={"Reward": lambda m: m.schedule.reward},
        )

    def _init_environment(self, env):
        # map
        for i, row in enumerate(env.grid):
            for j, cell in enumerate(row):
                if cell != "X":
                    agent = GridAgent(self.next_id(), self, symbol=cell)
                    self.grid.place_agent(agent, (i, j))
        for coords in env.targets:
            self.grid.place_agent(TargetAgent(self.next_id(), self), coords)

        # passengers
        pass_agent = PassengerAgent(
            self.next_id(), self, target_id=env.state["dest_idx"]
        )
        self.schedule.add(pass_agent)
        self.grid.place_agent(pass_agent, env.passenger_loc)

        # vehicle
        veh_agent = VehicleAgent(self.next_id(), self)
        self.schedule.add(veh_agent)
        self.grid.place_agent(veh_agent, env.vehicle_loc)

    def step(self):
        self.schedule.step()
        # if done:
        #     self.running = False
        self.datacollector.collect(self)
