from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import SmartCabModel
from .visualization import agent_portrayal, ResultsElement

CANVAS_HEIGHT = 500
CANVAS_WIDTH = 500
grid_h, grid_w = 8, 8

grid = CanvasGrid(agent_portrayal, grid_w, grid_h, CANVAS_WIDTH, CANVAS_HEIGHT)
chart = ChartModule(
    [{"Label": "Reward", "Color": "Green"}], data_collector_name="datacollector"
)
text = ResultsElement()

model_params = {
    "width": grid_w,
    "height": grid_h,
    "show_symbols": UserSettableParameter(
        "checkbox",
        "Show Directions",
        value=True,
    ),
    "nb_episodes": UserSettableParameter(
        "slider",
        "Number of Episodes",
        value=1,
        min_value=1,
        max_value=1000,
        step=1,
    ),
}

server = ModularServer(
    SmartCabModel, [grid, text], "SmartCab Model", model_params=model_params
)
