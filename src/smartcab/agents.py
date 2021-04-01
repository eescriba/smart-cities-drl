import itertools
from mesa import Agent as MesaAgent
from .enums import GridSymbol


class GridAgent(MesaAgent):
    def __init__(self, unique_id, model, symbol):
        self.symbol = GridSymbol(symbol)
        super().__init__(unique_id, model)


class VehicleAgent(MesaAgent):
    """ An agent """

    def __init__(self, unique_id, model):
        self.heading = "N"
        super().__init__(unique_id, model)

    def step(self, dest):
        self.heading = self.model.grid.get_heading(self.heading, self.pos, dest)
        self.model.grid.move_agent(self, dest)


class PassengerAgent(MesaAgent):
    """ An agent """

    def __init__(self, unique_id, model, target_id):
        self.target_id = target_id
        super().__init__(unique_id, model)

    def step(self, dest):
        self.model.grid.move_agent(self, dest)


class TargetAgent(MesaAgent):

    id_iter = itertools.count()

    def __init__(self, unique_id, model, dest=False):
        self.target_id = next(self.id_iter)
        self.dest = dest
        super().__init__(unique_id, model)