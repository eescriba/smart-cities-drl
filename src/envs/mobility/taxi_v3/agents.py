from mesa import Agent as MesaAgent

from core.utils import get_heading


class TaxiAgent(MesaAgent):
    """ An agent """

    def __init__(self, unique_id, model):
        self.heading = "N"
        super().__init__(unique_id, model)

    @property
    def portrayal(self):
        return {
            "Shape": "resources/taxi-" + self.heading + ".png",
            "h": 0.75,
            "w": 0.75,
            "Layer": 2,
        }

    def step(self, dest):
        self.heading = get_heading(self.heading, self.pos, dest)
        self.model.grid.move_agent(self, dest)


class PassengerAgent(MesaAgent):
    """ An agent """

    @property
    def portrayal(self):
        return {
            "Filled": "true",
            "Shape": "circle",
            "r": 0.5,
            "Layer": 3,
            "Color": "black",
        }

    def step(self, dest):
        self.model.grid.move_agent(self, dest)


class GridAgent(MesaAgent):
    def __init__(self, unique_id, model, wall=None):
        self.wall = wall
        super().__init__(unique_id, model)

    @property
    def portrayal(self):
        return {
            "Filled": "true",
            "Shape": "resources/bush-r.png" if self.wall else "rect",
            "h": 1,
            "w": 1,
            "Layer": 0,
            "Color": "#343a40",
        }


class LocationAgent(GridAgent):
    def __init__(self, unique_id, model, color, dest=False):
        self.dest = dest
        self.color = color
        super().__init__(unique_id, model)

    @property
    def portrayal(self):
        return {
            "Shape": "resources/" + self.color + ".png",
            "h": 1 if self.dest else 0.5,
            "w": 1 if self.dest else 0.5,
            "Layer": 1,
        }