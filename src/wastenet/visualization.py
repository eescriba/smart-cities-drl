from mesa.visualization.modules import TextElement


def network_portrayal(G):
    def node_color(agent):
        if not hasattr(agent, "fill_level"):
            return "black"
        if agent.fill_level >= 100:
            return "red"
        elif agent.fill_level >= 75:
            return "orange"
        elif agent.fill_level > 25:
            return "#ffe900"
        return "#008000"

    def node_size(agent):
        if hasattr(agent, "fill_level") and agent.fill_level == 0:
            return 8
        return 6

    def edge_color(agent1, agent2):
        if agent2.prev == agent1.unique_id or agent2.prev == agent1.unique_id:
            return "#000000"
        return "#e8e8e8"

    def get_agents(source, target):
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    def get_tooltip(agent):
        return (
            "id: {}<br>fill_level: {}%".format(agent.unique_id, agent.fill_level)
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


level_fields = [
    {"Label": "", "Color": "transparent"},
    {"Label": "Fill level (%)", "Color": "#83c3e3"},
]


class ResultsElement(TextElement):
    def render(self, model):
        env = model.env
        reward = model.schedule.reward
        last_reward = model.schedule.last_reward
        return f"<div > \
            <div style='float: left; width: 50%;'> \
                <h3>Episodes left: {model.remaining_episodes} </h3> \
                <h3>Current run</h3> \
                <span>Day: {env.current_day}</span><br> \
                <span>Node: {env.current_node}</span><br> \
                <span>Reward: {reward}({last_reward:+})</span><br> \
            </div> \
            <div style='float: left; width: 50%; margin-bottom: 20px;'> \
                <h3>Avg per route</h3> \
                <span>Distance: {model.mean_dist:.2f}</span><br> \
                <span>Collected: {model.mean_collected:.2f}</span><br> \
                <span>Overflows: {model.mean_overflow:.2f}</span><br> \
                <span>Reward: {model.mean_reward:.2f}</span> \
            </div> \
            </div>"
