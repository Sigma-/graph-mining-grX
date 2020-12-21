from load_data import load_sociopatterns_network, drawGraph
import networkx as nx
import collections
import matplotlib.pyplot as plt
import pandas as pd
from nams.functions import ecdf
from nxviz import CircosPlot, ArcPlot


G = load_sociopatterns_network()

def number_of_nodes_and_edges(G):
    number_of_nodes = len(G.nodes)
    number_of_edges = len(G.edges)
    print(f"# of Nodes: {number_of_nodes} \n# of Edges: {number_of_edges}")

def rank_ordered_neighbors(G):
    """
    Uses a pandas Series to help with sorting.
    """
    s = pd.Series({n: len(list(G.neighbors(n))) for n in G.nodes()})
    return s.sort_values(ascending=False)

orderded_neighbors = rank_ordered_neighbors(G)
#print(orderded_neighbors)

def degree_centrality_distribution(G):
    keysdc = nx.degree_centrality(G).keys()
    valuesdc = nx.degree_centrality(G).values()
    dictionarydc = dict(zip(keysdc, valuesdc))
    s = pd.Series(dictionarydc)
    return s.sort_values(ascending=False)

degree_centrality_distribution(G)

def list_node_neighbors(node_id):
    l = list(G.neighbors(node_id))
    print(f'The neighbors of the node # {node_id} are: \n {l}')
#list_node_neighbors(804)

def ecdf_degree_centrality(G):
    """ECDF of degree centrality."""
    x, y = ecdf(list(nx.degree_centrality(G).values()))
    plt.title("ECDF degree centralité distribution" )
    plt.scatter(x, y)
    plt.xlabel("degree centrality")
    plt.ylabel("cumulative fraction")
    markers = (0.5, 0.25, 0.75)
    for quartiles in markers:
        plt.axhline(quartiles, color='g')
        plt.legend(["Quartils"])
    plt.show()
# ecdf_degree_centrality(G)

def circos_plot(G):
    """Draw a Circos Plot of the graph."""
    c = CircosPlot(G, node_order="order", node_color="order", node_labels=True, node_size=200, fontsize=6 )
    c.draw()
    plt.show()
#circos_plot(G)

def arc_plot(G):
    """Draw an arc Plot of the graph."""
    a = ArcPlot(G, node_order="order", node_color="order", node_labels=True, fontsize=6)
    a.draw()
    plt.show()
#arc_plot(G)

#compute the differents centrality
cc = nx.closeness_centrality(G)
bc = nx.betweenness_centrality(G)
kc = nx.katz_centrality_numpy(G)
prc = nx.pagerank(G)
dc = nx.degree_centrality(G)


#list the graph nodes
Nodes = G.nodes()

clC = nx.closeness_centrality (G)
bC = nx.betweenness_centrality (G)
kC = nx.katz_centrality_numpy (G)
prC = nx.pagerank (G)
degC = nx.degree_centrality (G)

keys = G.nodes ()

correlation = [clC, bC, kC, prC, degC]
correlations = [[clC[k] for k in keys], [bC[k] for k in keys], [kC[k] for k in keys], [degC[k] for k in keys]]

names = ['Closeness', 'Betweenness', 'Katz', 'Degree']


c_max = max(correlations[0])
c_min = min(correlations[0])
b_max = max(correlations[1])
b_min = min(correlations[1])
k_max = max(correlations[2])
k_min = min(correlations[2])
d_max = max(correlations[3])
d_min = min(correlations[3])

def get_max(correlation):
    for i in range(len(correlation)):
        list_max = max(correlations[i])
        print(f"{names[i]} max: {list_max}")
    return list_max

def get_key(c_max):
    for i in correlation:
        for key, value in i.items():
            if c_max == value:
                print(f"initial {key}: {c_max}")
    return "key doesn't exist"

get_key(k_min)


#print(correlation)

#print(correlations[0])


plt.figure (figsize=(15,10))
for i in range (4):
    plt.plot (correlations[i], label = names[i])
    plt.legend ()
    plt.title(f"Distrution des différentes centralité: \n {names}")
plt.show ()

from scipy import stats

# for i in range (5):
#      for j in range (i + 1, 5):
#          print (names[i], ' vs ', names[j], ': PearsonResult', stats.pearsonr (correlations[i], correlations[j]), stats.spearmanr (correlations[i], correlations[j]))


# print(list(G.neighbors(272)))
# print(list(G.neighbors(210)))
# print(list(G.neighbors(791)))
# print(list(G.neighbors(87)))
# print(list(G.neighbors(804)))
print(nx.average_clustering(G))