import random

from core.scheduler import RLlibActivation
from .agents import DumpsterAgent, BaseAgent
from .enums import WasteNetAction, WasteNetMode


class WasteNetActivation(RLlibActivation):
    """
    Scheduler for WasteNet env.
    """

    PARTIAL_TH = 40

    def __init__(self, model, mode, rl_agent=None) -> None:
        super().__init__(model, rl_agent)
        self.mode = mode
        self.last_checkpoint = 0
        self.mode_actions = {
            WasteNetMode.COMPLETE.name: lambda: WasteNetAction.PICKUP,
            WasteNetMode.PARTIAL.name: self._next_action_th,
            WasteNetMode.RANDOM.name: self._next_action_rnd,
        }

    def step(self) -> None:

        action = self.mode_actions.get(self.mode, self.next_action)()
        done = self.forward(action)

        agent_idx = self.model.env.current_node
        if agent_idx == self.model.env.start_node:
            for agent in self.agents:
                agent.checkpoint = False
                agent.prev = -1
                if isinstance(agent, DumpsterAgent):
                    agent.fill_level = self.model.env.fill_levels[agent.unique_id - 1]
            self.agents[agent_idx].checkpoint = True
        elif agent_idx == self.model.env.end_node:
            self._update_path(agent_idx)
            self.last_checkpoint = self.model.env.start_node
        elif action == WasteNetAction.PICKUP:
            self._update_path(agent_idx)
            self.last_checkpoint = agent_idx
            self.agents[agent_idx].fill_level = 0
            self.agents[agent_idx].checkpoint = True

        self.steps += 1
        self.time += 1

        return done

    def _next_action_rnd(self):
        return random.choice(list(WasteNetAction))

    def _next_action_th(self):
        node = self.model.env.current_node
        if node >= self.model.env.nb_dumpsters:
            return WasteNetAction.AVOID
        return (
            WasteNetAction.PICKUP
            if self.model.env.fill_levels[self.model.env.current_node] > self.PARTIAL_TH
            else WasteNetAction.AVOID
        )

    def _update_path(self, agent_idx):
        path = self.model.env.current_path
        for i, node in enumerate(path):
            if i == 0:
                continue
            self.agents[node].prev = self.agents[path[i - 1]].unique_id
