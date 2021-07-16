from core.scheduler import RLlibActivation

from .agents import PassengerAgent, VehicleAgent


class SmartCabActivation(RLlibActivation):
    """
    A scheduler which activates each agent once per step, in random order.
    """

    def step(self) -> None:
        """
        Executes the step of all agents, one at a time, in random order.
        """
        action = self.next_action()
        done = self.forward(action)

        vehicle_loc = self.model.env.vehicle_loc
        targets = self.model.env.targets
        pass_idx = self.model.env.state["pass_idx"]

        for agent in self.agent_buffer(shuffled=True):
            if isinstance(agent, VehicleAgent):
                agent.step(vehicle_loc)
            elif isinstance(agent, PassengerAgent):
                coords = vehicle_loc if pass_idx == len(targets) else targets[pass_idx]
                agent.step(coords)

        self.steps += 1
        self.time += 1

        return done
