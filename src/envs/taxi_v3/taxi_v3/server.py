from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .env import env
from .model import TaxiModel
from .visualization import agent_portrayal


CANVAS_HEIGHT = 500
CANVAS_WIDTH = 500
grid_h, grid_w = env.shape

grid = CanvasGrid(agent_portrayal, grid_h, grid_w, CANVAS_HEIGHT, CANVAS_WIDTH)
chart = ChartModule(
    [{"Label": "Reward", "Color": "Green"}], data_collector_name="datacollector"
)

model_params = {
    "env": UserSettableParameter(
        "choice",
        "Environment",
        value="Taxi-v3",
        choices=["Taxi-v3"],
    ),
    "stage": UserSettableParameter(
        "choice",
        "Stage",
        value="Test",
        choices=["Test", "Train"],
    ),
    "rl_agent": UserSettableParameter(
        "choice",
        "RL Agent",
        value="DQN",
        choices=["DQN"],
    ),
}

server = ModularServer(
    TaxiModel, [grid, chart], "Taxi Model DRL", model_params=model_params
)
