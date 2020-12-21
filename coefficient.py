from load_data import load_sociopatterns_network, drawGraph
import networkx as nx

G = load_sociopatterns_network()

print(nx.average_clustering(G))