
from copy import deepcopy

from mesa import Agent as MesaAgent
from rl.agents.dqn import DQNAgent


class TaxiAgent(MesaAgent):
    """ An agent """        

    def step(self, dest):
        self.model.grid.move_agent(self, dest)


class PassengerAgent(MesaAgent):
    """ An agent """
    def step(self, dest):
        self.model.grid.move_agent(self, dest)

class LocationAgent(MesaAgent):
    def __init__(self, unique_id, model, color, dest=False):
        self.color = color
        self.dest = dest
        super().__init__(unique_id, model)

class WallAgent(MesaAgent):
    """ An agent """
    pass

class LocationWallAgent(LocationAgent, WallAgent):
    """ An agent """
    pass
