from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from taxi_ql.agents import TaxiAgent, PassengerAgent, LocationAgent
from taxi_ql.model import TaxiModel


def agent_portrayal(agent):
    portrayal = {"Filled": "true"}

    if type(agent) is TaxiAgent:
        # portrayal["Shape"] = "resources/taxi.png"
        # portrayal["Shape"] = "https://icons.iconarchive.com/icons/icons-land/transporter/64/Taxi-Top-Yellow-icon.png"
        portrayal["Shape"] = "rect"
        portrayal["Color"] = "black"
        portrayal["Layer"] = 1
        portrayal["h"] = 0.75
        portrayal["w"] = 0.75
       
    elif type(agent) is PassengerAgent:
        portrayal["Layer"] = 2
        portrayal["Color"] = "grey"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5

    elif type(agent) is LocationAgent:
        portrayal["Layer"] = 0
        portrayal["Shape"] = "rect"
        portrayal["h"] = 1
        portrayal["w"] = 1
        portrayal["Color"] = "green"

    return portrayal


grid = CanvasGrid(agent_portrayal, 5, 5, 500, 500)

server = ModularServer(
    TaxiModel, [grid], "Taxi Model", {"N": 100, "width": 5, "height": 5}
)

server.port = 8521  # The default