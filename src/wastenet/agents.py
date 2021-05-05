from mesa import Agent


class DumpsterAgent(Agent):
    def __init__(self, unique_id, model, fill_level):
        super().__init__(unique_id, model)
        self.fill_level = fill_level
        self.checkpoint = False
        self.prev = -1

    def step(self):
        pass


class BaseAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.checkpoint = True
        self.prev = -1

    def step(self):
        pass
