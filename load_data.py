import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import collections


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

def drawGraph (G, pos = None):
    plt.figure (figsize=(20,20))
    if not pos:
        pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos) 
    plt.show()
    

G = load_sociopatterns_network()
drawGraph(G)