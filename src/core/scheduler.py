from mesa.time import BaseScheduler


class RLKerasActivation(BaseScheduler):
    """
    Keras-RL agent scheduler
    """

    def __init__(self, model, rl_agent) -> None:
        super().__init__(model)
        self.rl_agent = rl_agent
        self.reward = 0

    def forward(self):
        """
        Execute next action
        """
        action = self.rl_agent.forward(self.model.env.s)
        observation, reward, done, info = self.model.env.step(action)
        print(observation, reward, done, info)
        self.model.env.s = observation
        self.reward += reward
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


class RLlibActivation(BaseScheduler):
    """
    RLlib agent scheduler
    """

    def __init__(self, model, rl_agent) -> None:
        super().__init__(model)
        self.rl_agent = rl_agent
        self.reward = 0
        self.last_reward = 0

    def forward(self):
        """
        Execute next action
        """
        action = self.rl_agent.compute_action(self.model.env.s)
        state, reward, done, info = self.model.env.step(action)
        print(state, reward, done, info)
        self.last_reward = reward
        self.reward += reward
        if done:
            self.model.env.reset()
            self.reward = 0
        return action, done

    def step(self) -> None:
        """
        Executes the step of all agents, one at a time, in random order.
        """
        self.forward()

        for agent in self.agent_buffer(shuffled=True):
            agent.step()

        self.steps += 1
        self.time += 1
