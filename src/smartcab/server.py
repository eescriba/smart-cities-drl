from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import SmartCabModel
from .visualization import agent_portrayal

# from .rl_agents import dqn


CANVAS_HEIGHT = 500
CANVAS_WIDTH = 500
grid_h, grid_w = 15, 15

grid = CanvasGrid(agent_portrayal, grid_w, grid_h, CANVAS_WIDTH, CANVAS_HEIGHT)
chart = ChartModule(
    [{"Label": "Reward", "Color": "Green"}], data_collector_name="datacollector"
)

model_params = {
    "width": grid_w,
    "height": grid_h,
    "show_symbols": UserSettableParameter(
        "checkbox",
        "Show Directions",
        value=True,
    ),
}

server = ModularServer(
    SmartCabModel, [grid], "SmartCab Model", model_params=model_params
)
