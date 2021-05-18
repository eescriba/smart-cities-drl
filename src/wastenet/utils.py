import networkx as nx


# def generate_graph():
#     G = nx.Graph()
#     G.add_edge(0, 1, weight=2)
#     G.add_edge(0, 2, weight=2)
#     G.add_edge(1, 2, weight=3)
#     G.add_edge(1, 3, weight=2)
#     G.add_edge(1, 4, weight=3)
#     G.add_edge(2, 3, weight=2)
#     G.add_edge(2, 5, weight=3)
#     G.add_edge(3, 4, weight=2)
#     G.add_edge(3, 5, weight=2)
#     G.add_edge(4, 5, weight=3)
#     G.add_edge(4, 6, weight=2)
#     G.add_edge(5, 6, weight=2)

#     G.add_edge(4, 7, weight=3)
#     G.add_edge(5, 8, weight=3)
#     G.add_edge(6, 7, weight=2)
#     G.add_edge(6, 8, weight=2)
#     G.add_edge(7, 8, weight=3)
#     G.add_edge(7, 9, weight=2)
#     G.add_edge(8, 9, weight=2)
#     return G


def generate_graph(nb_nodes=10):
    assert nb_nodes % 3 == 1, "Invalid number of nodes"
    G = nx.Graph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(0, 2, weight=2)
    G.add_edge(1, 2, weight=3)
    G.add_edge(1, 3, weight=2)
    G.add_edge(2, 3, weight=2)
    for j in range(nb_nodes // 3 - 1):
        i = j * 3
        G.add_edge(i + 1, i + 4, weight=3)
        G.add_edge(i + 2, i + 5, weight=3)
        G.add_edge(i + 3, i + 4, weight=2)
        G.add_edge(i + 3, i + 5, weight=2)
        G.add_edge(i + 4, i + 5, weight=3)
        G.add_edge(i + 4, i + 6, weight=2)
        G.add_edge(i + 5, i + 6, weight=2)
    return G


def generate_fill_ranges():
    # Mean: 40
    return [
        (20, 40),
        (30, 70),
        (40, 90),
        (10, 50),
        (20, 30),
        (10, 20),
        (40, 80),
        (30, 60),
    ]
