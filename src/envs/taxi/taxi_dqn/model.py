import random
from copy import deepcopy

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
# from mesa.time import RandomActivation

from taxi_dqn.agents import TaxiAgent, PassengerAgent, LocationAgent
from taxi_dqn.scheduler import RLActivation


class TaxiModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height, rl_agent, env):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RLActivation(self, rl_agent)
        self.running = True

        taxi = TaxiAgent(4, self)
        self.schedule.add(taxi)
        taxi_x = self.random.randrange(self.grid.width)
        taxi_y = self.random.randrange(self.grid.height)
        self.grid.place_agent(taxi, (taxi_x, taxi_y))

        self.locations = [(0,0), (0,4), (3,0), (4,4)]
        locs_colors = ['#e6c800', 'red', '#00afff', 'green']

        pass_index = self.random.randrange(len(self.locations))
        dest_index = self.random.choice([i for i in range(0,4) if i != pass_index])

        for i in range(0, 4):
            loc= LocationAgent(i, self, color=locs_colors[i], dest= (i==dest_index) )
            self.schedule.add(loc)
            self.grid.place_agent(loc, self.locations[i])

        passenger = PassengerAgent(5, self)
        self.schedule.add(passenger)
        self.grid.place_agent(passenger, self.locations[pass_index])
        env.reset()
        # (taxi row, taxi column, passenger index, destination index)
        state = env.encode(taxi_x, taxi_y, pass_index, dest_index) 
        env.s = state
        self.env = env
        

    def step(self):
        self.schedule.step()