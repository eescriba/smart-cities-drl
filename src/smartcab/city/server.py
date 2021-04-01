from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from smartcab.env import SmartCabEnv
from smartcab.model import MobilityModel
from smartcab.visualization import agent_portrayal

from .rl_agents import dqn


CANVAS_HEIGHT = 500
CANVAS_WIDTH = 500
grid_h, grid_w = 25, 25

grid = CanvasGrid(agent_portrayal, grid_w, grid_h, CANVAS_WIDTH, CANVAS_HEIGHT)
chart = ChartModule(
    [{"Label": "Reward", "Color": "Green"}], data_collector_name="datacollector"
)


model_params = {
    # "env": UserSettableParameter(
    #     "choice",
    #     "Environment",
    #     value="SmartCab",
    #     choices=["SmartCab", "WasteNet"],
    # ),
    # "rl_agent": UserSettableParameter(
    #     "choice",
    #     "RL Agent",
    #     value="DQN",
    #     choices=["DQN", "CEM", "SARSA"],
    # ),
    "env": SmartCabEnv(file_path="../resources/city25.txt"),
    "rl_agent": dqn,
    "width": grid_w,
    "height": grid_h,
    "show_symbols": UserSettableParameter(
        "checkbox",
        "Show Directions",
        value=True,
    ),
    "nb_vehicles": UserSettableParameter("slider", "Number of vehicles", 1, 1, 2),
    "nb_passengers": UserSettableParameter("slider", "Number of passengers", 1, 1, 4),
    "nb_targets": UserSettableParameter("slider", "Number of targets", 4, 4, 4),
}

server = ModularServer(
    MobilityModel, [grid], "Mobility Model DRL", model_params=model_params
)
