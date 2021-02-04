from mesa import Model

from mesa.datacollection import DataCollector

# from mesa.time import RandomActivation

from .agents import TaxiAgent, PassengerAgent, LocationAgent, GridAgent
from .env import MesaTaxiEnv
from .rl_agents import dqn
from .scheduler import TaxiActivation
from .space import TaxiMultiGrid


class TaxiModel(Model):

    env_choices = {"Taxi-v3": MesaTaxiEnv}
    rl_choices = {"DQN": dqn}

    def __init__(self, rl_agent, env, stage, width=5, height=5):

        super().__init__()

        if isinstance(env, str):
            env = self.env_choices.get(env)()
        if isinstance(rl_agent, str):
            rl_agent = self.rl_choices.get(rl_agent)

        self.grid = TaxiMultiGrid(width, height, True)
        self.schedule = TaxiActivation(self, rl_agent)

        self.done = False
        self.datacollector = DataCollector(
            model_reporters={"Reward": lambda m: m.schedule.reward},
        )

        taxi_coords = self.random.randrange(width), self.random.randrange(height)
        pass_index = self.random.randrange(len(env.locs))
        dest_index = self.random.choice(
            [i for i in range(0, len(env.locs)) if i != pass_index]
        )

        self._init_taxi(taxi_coords)
        self._init_locations(env, dest_index)
        self._init_passenger(env, pass_index)
        env.reset()
        self._init_grid(env)
        # (taxi row, taxi column, passenger index, destination index)
        state = env.encode(taxi_coords[0], taxi_coords[1], pass_index, dest_index)
        env.s = state
        self.env = env

    def _init_taxi(self, coords):
        taxi = TaxiAgent(self.next_id(), self)
        self.schedule.add(taxi)
        self.grid.place_agent(taxi, coords)

    def _init_locations(self, env, dest_index):
        for i in range(0, len(env.locs)):
            loc = LocationAgent(
                unique_id=self.next_id(),
                model=self,
                dest=(i == dest_index),
                color=env.locs_colors[i],
            )
            self.grid.place_agent(loc, env.locs[i])

    def _init_passenger(self, env, pass_index):
        passenger = PassengerAgent(self.next_id(), self)
        self.schedule.add(passenger)
        self.grid.place_agent(passenger, env.locs[pass_index])

    def _init_grid(self, env):
        for i, row in enumerate(env.cells):
            for j, _ in enumerate(row):
                if row[j] == b"|" or row[j] == b":":
                    continue
                if j < len(row) - 1 and row[j + 1] == b"|":
                    agent = GridAgent(self.next_id(), self, wall=True)
                    self.grid.place_agent(agent, (i, int(j / 2)))

                agent = GridAgent(self.next_id(), self)
                self.grid.place_agent(agent, (i, int(j / 2)))

    def step(self):
        done = self.schedule.step()
        if done:
            self.running = False
        self.datacollector.collect(self)
