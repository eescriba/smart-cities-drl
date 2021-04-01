from core.scheduler import RLActivation

from .agents import PassengerAgent, VehicleAgent


class SmartCabActivation(RLActivation):
    """
    A scheduler which activates each agent once per step, in random order.
    """

    def step(self) -> None:
        """
        Executes the step of all agents, one at a time, in random order.
        """

        done = self.forward()
        taxi_x, taxi_y, pass_idx, dest_idx = self.model.state
        for agent in self.agent_buffer(shuffled=True):
            if isinstance(agent, VehicleAgent):
                agent.step((taxi_x, taxi_y))
            elif isinstance(agent, PassengerAgent):
                coords = (
                    (taxi_x, taxi_y)
                    if pass_idx == 4
                    else self.model.env.targets[pass_idx]
                )
                agent.step(coords)

        self.steps += 1
        self.time += 1

        return done