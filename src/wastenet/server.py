import math

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import BarChartModule, ChartModule, NetworkModule
from .model import WasteNet
from .enums import WasteNetMode
from .visualization import network_portrayal, level_series, level_fields, ResultsElement


network = NetworkModule(network_portrayal, 400, 500, library="d3")
# chart = ChartModule(level_series)
barchart = BarChartModule(level_fields, canvas_width=500, scope="agent")
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
    "nb_episodes": UserSettableParameter(
        "slider",
        "Number of Episodes",
        value=1,
        min_value=1,
        max_value=2000,
        step=1,
    ),
}

server = ModularServer(
    WasteNet, [network, text, barchart], "WasteNet Model", model_params
)
server.port = 8521
