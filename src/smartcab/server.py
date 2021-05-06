from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import MobilityModel
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
    "rl_agent": None,
    "width": grid_w,
    "height": grid_h,
    "show_symbols": UserSettableParameter(
        "checkbox",
        "Show Directions",
        value=True,
    ),
    "nb_vehicles": UserSettableParameter("slider", "Number of vehicles", 1, 1, 2),
    "nb_passengers": UserSettableParameter("slider", "Number of passengers", 1, 1, 4),
    "nb_targets": UserSettableParameter("slider", "Number of targets", 6, 6, 6),
}

server = ModularServer(
    MobilityModel, [grid], "Mobility Model DRL", model_params=model_params
)
