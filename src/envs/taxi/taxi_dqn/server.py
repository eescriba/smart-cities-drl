from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from taxi_dqn.model import TaxiModel
from taxi_dqn.dqn import dqn, env
from taxi_dqn.visualization import agent_portrayal

grid = CanvasGrid(agent_portrayal, 5, 5, 500, 500)

server = ModularServer(
    TaxiModel, [grid], "Taxi Model", {"N": 100, "width": 5, "height": 5, "rl_agent": dqn, "env": env}
)

server.port = 8521  # The default