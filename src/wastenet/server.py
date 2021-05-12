import math

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from .model import WasteNet
from .enums import WasteNetMode
from .visualization import network_portrayal, level_series, ResultsElement


network = NetworkModule(network_portrayal, 500, 500, library="d3")
chart = ChartModule(level_series)
text = ResultsElement()


model_params = {
    # "nb_nodes": UserSettableParameter(
    #     "slider",
    #     "Number of nodes",
    #     value=9,
    #     min_value=6,
    #     max_value=12,
    #     step=3,
    # ),
    "mode": UserSettableParameter(
        "choice",
        "Agent Mode",
        value=WasteNetMode.COMPLETE.name,
        choices=WasteNetMode.names(),
    ),
    "nb_days": UserSettableParameter(
        "slider",
        "Total Days",
        value=15,
        min_value=7,
        max_value=30,
        step=1,
    ),
}

server = ModularServer(WasteNet, [network, text, chart], "WasteNet Model", model_params)
server.port = 8521
