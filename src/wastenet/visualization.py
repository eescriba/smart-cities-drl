from mesa.visualization.modules import TextElement


def network_portrayal(G):
    def node_color(agent):
        if not hasattr(agent, "fill_level"):
            return "black"
        if agent.fill_level >= 1.0:
            return "red"
        elif agent.fill_level >= 0.8:
            return "orange"
        elif agent.fill_level > 0.2:
            return "#ffe900"
        return "#008000"

    def node_size(agent):
        if hasattr(agent, "fill_level") and agent.fill_level == 0:
            return 8
        return 6

    def edge_color(agent1, agent2):
        if agent2.prev == agent1.unique_id or agent2.prev == agent1.unique_id:
            # if agent1.checkpoint and agent2.checkpoint:
            return "#000000"
        return "#e8e8e8"

    # def edge_width(agent1, agent2):
    #     if agent1.checkpoint and agent2.checkpoint:
    #         return 3
    #     return 2

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    def get_tooltip(agent):
        return (
            "id: {}<br>fill_level: {}".format(agent.unique_id, agent.fill_level)
            if hasattr(agent, "fill_level")
            else "Base"
        )

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": node_size(agents[0]),
            "color": node_color(agents[0]),
            "tooltip": get_tooltip(agents[0]),
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": 2,
        }
        for (source, target) in G.edges
    ]

    return portrayal


level_series = [
    {"Label": "Empty", "Color": "#008000"},
    {"Label": "Medium", "Color": "yellow"},
    {"Label": "Full", "Color": "orange"},
    {"Label": "Overflow", "Color": "red"},
]


class ResultsElement(TextElement):
    def render(self, model):
        day = model.env.current_day
        node = model.env.current_node
        reward = model.schedule.reward
        last_reward = model.schedule.last_reward
        # return "Day: {}<br>Node: {}<br>Reward: {}({0:+d})".format(
        #     day, node, reward, last_reward
        # )
        return f"Day: {day} <br>Node: {node}<br>Reward: {reward}({last_reward:+})"
