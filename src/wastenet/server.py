import math

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from .model import WasteNet
from .visualization import network_portrayal, level_series, ResultsElement


network = NetworkModule(network_portrayal, 500, 500, library="d3")
chart = ChartModule(level_series)
text = ResultsElement()


model_params = {
    "nb_nodes": UserSettableParameter(
        "slider",
        "Number of nodes",
        7,
        7,
        10,
        1,
        description="Choose how many nodes to include in the model",
    ),
}

server = ModularServer(WasteNet, [network, text, chart], "WasteNet Model", model_params)
server.port = 8521
