import random
from copy import deepcopy

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

# from mesa.time import RandomActivation

from taxi_dqn.agents import TaxiAgent, PassengerAgent, LocationAgent
from taxi_dqn.scheduler import TaxiActivation


class TaxiModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height, rl_agent, env):
        self.grid = MultiGrid(width, height, True)
        self.schedule = TaxiActivation(self, rl_agent)
        self.running = True

        taxi = TaxiAgent(4, self)
        self.schedule.add(taxi)
        taxi_x = self.random.randrange(self.grid.width)
        taxi_y = self.random.randrange(self.grid.height)
        self.grid.place_agent(taxi, (taxi_x, taxi_y))


        locs_colors = ["#e6c800", "red", "#00afff", "green"]
        pass_index = self.random.randrange(len(env.locs))
        dest_index = self.random.choice(
            [i for i in range(0, len(env.locs)) if i != pass_index]
        )

        for i in range(0, 4):
            loc = LocationAgent(i, self, color=locs_colors[i], dest=(i == dest_index))
            self.schedule.add(loc)
            self.grid.place_agent(loc, env.locs[i])

        passenger = PassengerAgent(5, self)
        self.schedule.add(passenger)
        self.grid.place_agent(passenger, env.locs[pass_index])
        env.reset()
        # (taxi row, taxi column, passenger index, destination index)
        state = env.encode(taxi_x, taxi_y, pass_index, dest_index)
        env.s = state
        self.env = env

    def step(self):
        self.schedule.step()