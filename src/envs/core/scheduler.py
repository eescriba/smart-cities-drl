from mesa.time import BaseScheduler


class RLActivation(BaseScheduler):
    """
    A scheduler which activates each agent once per step, in random order,
    """

    def __init__(self, model, rl_agent) -> None:
        super().__init__(model)
        self.rl_agent = rl_agent

    def forward(self):
        """
        Execute next action
        """
        action = self.rl_agent.forward(self.model.env.s)
        observation, reward, done, info = self.model.env.step(action)
        print(observation, reward, done, info)
        self.model.env.s = observation
        self.model.reward = reward
        if done:
            self.rl_agent.forward(observation)
            self.rl_agent.backward(0.0, terminal=False)
        return done

    def step(self) -> None:
        """
        Executes the step of all agents, one at a time, in random order.
        """
        self.forward()

        for agent in self.agent_buffer(shuffled=True):
            agent.step()

        self.steps += 1
        self.time += 1
