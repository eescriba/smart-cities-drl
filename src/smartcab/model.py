from mesa import Model
from mesa.datacollection import DataCollector
from core.rl import PPOAgent

from .agents import GridAgent, PassengerAgent, TargetAgent, VehicleAgent
from .env import SmartCabEnv
from .ppo import best_config
from .scheduler import SmartCabActivation
from .space import SmartCabMultiGrid


class SmartCabModel(Model):
    def __init__(self, show_symbols, nb_episodes=1, width=8, height=8):

        super().__init__()

        self.grid = SmartCabMultiGrid(width, height, True)
        self.show_symbols = show_symbols

        env_config = {}
        self.env = SmartCabEnv(env_config)
        rl_agent = PPOAgent("SmartCab", SmartCabEnv, env_config, best_config)
        rl_agent.load("./checkpoints/checkpoint-best")

        self.schedule = SmartCabActivation(self, rl_agent=rl_agent)
        self._init_environment(self.env)

        self.datacollector = DataCollector(
            model_reporters={"Reward": lambda m: m.schedule.reward},
        )
        self.remaining_episodes = nb_episodes

    def _init_environment(self, env):
        for i, row in enumerate(env.grid):
            for j, cell in enumerate(row):
                if cell != "X":
                    agent = GridAgent(self.next_id(), self, symbol=cell)
                    self.grid.place_agent(agent, (i, j))
        for coords in env.targets:
            self.grid.place_agent(TargetAgent(self.next_id(), self), coords)

        # passengers
        pass_agent = PassengerAgent(
            self.next_id(), self, target_id=env.state["dest_idx"]
        )
        self.schedule.add(pass_agent)
        self.grid.place_agent(pass_agent, env.passenger_loc)

        # vehicle
        veh_agent = VehicleAgent(self.next_id(), self)
        self.schedule.add(veh_agent)
        self.grid.place_agent(veh_agent, env.vehicle_loc)

    def step(self):
        done = self.schedule.step()
        if done:
            self.env.reset()
            self.remaining_episodes -= 1
        if self.remaining_episodes == 0:
            self.running = False

        self.datacollector.collect(self)
