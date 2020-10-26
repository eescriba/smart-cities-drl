from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from taxi_dqn.agents import TaxiAgent, PassengerAgent, LocationAgent


class TaxiModel(Model):
    """A model with some number of agents."""
    def __init__(self, N, width, height):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.running = True

        taxi = TaxiAgent(4, self)
        self.schedule.add(taxi)
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(taxi, (x, y))

        passenger = PassengerAgent(5, self)
        self.schedule.add(passenger)
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(passenger, (x, y))

        locs_coords = [(0,0), (0,4), (3,0), (4,4)]
        locs_colors = ['#e6c800', 'red', '#00afff', 'green']

        for i in range(0, 4):
            loc= LocationAgent(i, self, locs_colors[i])
            self.schedule.add(loc)
            self.grid.place_agent(loc, locs_coords[i])
        

    def step(self):
        self.schedule.step()