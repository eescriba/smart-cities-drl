import random
from collections import namedtuple
from itertools import repeat

from mesa import Model
from mesa.datacollection import DataCollector

# from mesa.time import RandomActivation

from .agents import GridAgent, PassengerAgent, TargetAgent, VehicleAgent
from .env import SmartCabEnv


from .scheduler import SmartCabActivation
from .space import MobilityMultiGrid


class MobilityModel(Model):
    def __init__(
        self,
        rl_agent,
        env,
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
        self._init_environment(env, nb_targets)
        self._init_vehicles(nb_vehicles)

        # self.done = False
        self.datacollector = DataCollector(
            model_reporters={"Reward": lambda m: m.schedule.reward},
        )

        env.reset()
        # state = env.encode(taxi_coords[0], taxi_coords[1], pass_index, dest_index)
        # env.s = state
        self.env = env

    def _init_environment(self, env, nb_targets):
        targets = set()
        Target = namedtuple("Target", "agent coords")
        for i, row in enumerate(env.grid):
            for j, cell in enumerate(row):
                coords = (i, j)
                if cell == "X":
                    targets.add(Target(TargetAgent(self.next_id(), self), coords))
                else:
                    agent = GridAgent(self.next_id(), self, symbol=cell)
                    self.grid.place_agent(agent, coords)
        targets = random.sample(targets, nb_targets)
        for target in targets:
            self.grid.place_agent(target.agent, target.coords)

        # passenger
        src, dst = random.sample(targets, 2)
        agent = PassengerAgent(self.next_id(), self, target_id=dst.agent.target_id)
        self.schedule.add(agent)
        self.grid.place_agent(agent, src.coords)

    def _init_vehicles(self, nb_vehicles):
        for _ in repeat(None, nb_vehicles):
            agent = VehicleAgent(self.next_id(), self)
            coords = self.random.randrange(self.grid.width), self.random.randrange(
                self.grid.height
            )
            self.schedule.add(agent)
            self.grid.place_agent(agent, coords)

    def step(self):
        self.schedule.step()
        # if done:
        #     self.running = False
        self.datacollector.collect(self)

    @property
    def state(self):
        return self.env.decode(self.env.s)
