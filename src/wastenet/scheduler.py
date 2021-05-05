import networkx as nx

from core.scheduler import RLlibActivation
from .agents import DumpsterAgent, BaseAgent


class WasteNetActivation(RLlibActivation):
    """
    RLlib scheduler for WasteNet env.
    """

    def step(self) -> None:

        action, done = self.forward()

        agent_idx = self.model.env.current_node

        if agent_idx == 0:
            for agent in self.agents:
                if isinstance(agent, DumpsterAgent):
                    agent.checkpoint = False
                    agent.prev = -1
                    agent.fill_level = self.model.env.fill_levels[agent.unique_id - 1]

        else:
            if action != 0:
                last_checkpoint = next(
                    a.unique_id
                    for a in reversed(self.agents[:agent_idx])
                    if a.checkpoint
                )
                path = nx.shortest_path(
                    self.model.G, source=last_checkpoint, target=agent_idx
                )
                for i, node in enumerate(path):
                    if i == 0:
                        continue
                    self.agents[node].prev = self.agents[path[i - 1]].unique_id
                self.agents[agent_idx].fill_level = 0.0
                self.agents[agent_idx].checkpoint = True

        self.steps += 1
        self.time += 1

        return done
