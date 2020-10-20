
from mesa import Agent


class TaxiAgent(Agent):
    """ An agent """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()


class PassengerAgent(Agent):
    """ An agent """
    pass

class LocationAgent(Agent):
    def __init__(self, unique_id, model, color):
        self.color = color
        super().__init__(unique_id, model)

class WallAgent(Agent):
    """ An agent """
    pass

class LocationWallAgent(LocationAgent, WallAgent):
    """ An agent """
    pass
