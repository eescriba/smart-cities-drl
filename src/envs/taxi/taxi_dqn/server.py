from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from taxi_dqn.agents import TaxiAgent, PassengerAgent, LocationAgent
from taxi_dqn.model import TaxiModel
from taxi_dqn.dqn import dqn, env

def agent_portrayal(agent):
    portrayal = {"Filled": "true"}

    if type(agent) is TaxiAgent:
        portrayal["Shape"] = "resources/taxi.png"
        portrayal["Layer"] = 1
        portrayal["h"] = 0.75
        portrayal["w"] = 0.75
       
    elif type(agent) is PassengerAgent:
        portrayal["Layer"] = 2
        portrayal["Color"] = "black"
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5

    elif type(agent) is LocationAgent:
        portrayal["Layer"] = 0
        portrayal["Shape"] = "rect"
        portrayal["h"] = 1
        portrayal["w"] = 1
        portrayal["Color"] = agent.color
        if agent.dest:
            portrayal["text"] = 'X'
            portrayal["text_color"] = "black"

    return portrayal


grid = CanvasGrid(agent_portrayal, 5, 5, 500, 500)

server = ModularServer(
    TaxiModel, [grid], "Taxi Model", {"N": 100, "width": 5, "height": 5, "rl_agent": dqn, "env": env}
)

server.port = 8521  # The default