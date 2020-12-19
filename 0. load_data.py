import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


df = pd.read_csv("tij_InVS.dat", sep= " ", header=None)
df["weight"] = 1 
df.columns = ["t", "i", "j", "weight"]

print(df["i"][0])

def load_sociopatterns_network():
    df = pd.read_csv("tij_InVS.dat", sep= " ", header=None)
    df["weight"] = 1 
    df.columns = ["t", "i", "j", "weight"]
    
    G = nx.Graph()
    for row in df.iterrows():
        p1 = row[1]["i"]
        p2 = row[1]["j"]
        if G.has_edge(p1, p2):
            G.edges[p1, p2]["weight"] += 1
        else:
            G.add_edge(p1, p2, weight=1)

    for n in sorted(G.nodes()):
        G.nodes[n]["order"] = float(n)
    
    return G

G = load_sociopatterns_network()
nx.draw(G)

print(len(G.nodes()), len(G.edges()))


# compute the differents centrality
cc = nx.closeness_centrality(G)
bc = nx.betweenness_centrality(G)
kc = nx.katz_centrality_numpy(G)
prc = nx.pagerank(G)
dc = nx.degree_centrality(G)


# list the graph nodes
Nodes = G.nodes()
print(Nodes)
#G = nx.convert_node_labels_to_integers(G)
Nodes2 = G.nodes()
print()
nx.draw(G)


correlations = [[cc[k] for k in Nodes], 
                [bc[k] for k in Nodes], 
                [kc[k] for k in Nodes], 
                [prc[k] for k in Nodes],
                [dc[k] for k in Nodes]]

names = ['Closeness', 'Betweenness', 'Katz', 'Page rank', 'Degree']


# create a graph for each centrality and display it
for i in range(len(names)):
    name = names[i]
    plt.title(name)
    plt.plot(correlations[i], label=names[i], color=(0, 0, 0, 1))
    plt.bar(Nodes, correlations[i], color=(0.5, 0.5, 0.5, 1))
    plt.xlabel("Nodes")
    plt.ylabel("Centrality")
    plt.show()