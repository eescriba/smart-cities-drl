from core.scheduler import RLActivation
from .agents import TaxiAgent, PassengerAgent


class TaxiActivation(RLActivation):
    """
    A scheduler which activates each agent once per step, in random order.
    """

    def step(self) -> None:
        """
        Executes the step of all agents, one at a time, in random order.
        """

        done = self.forward()
        taxi_x, taxi_y, pass_index, dest_index = self.model.env.decode(self.model.env.s)

        for agent in self.agent_buffer(shuffled=True):
            if isinstance(agent, TaxiAgent):
                agent.step((taxi_x, taxi_y))
            elif isinstance(agent, PassengerAgent):
                coords = (
                    (taxi_x, taxi_y)
                    if pass_index == 4
                    else self.model.env.locs[pass_index]
                )
                agent.step(coords)

        self.steps += 1
        self.time += 1

        return done