#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


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
    nx.draw_networkx_labels(G, pos, alpha=0.1) 
    plt.show()
    

G = load_sociopatterns_network()
drawGraph(G)

#j'ai changé le Alpha 


# # Methode Louvain

# In[11]:


import community
import matplotlib as mpl

def partition_and_draw (G):
    pos = nx.spring_layout(G)
    partition = community.best_partition(G)
    size = float(len(set(partition.values())))
    count = 0.
    plt.figure(figsize=(14.0, 14.0))
    cmap = plt.get_cmap("tab10")
    cm = plt.cm.ScalarMappable(cmap = plt.get_cmap('Paired'), 
                                norm = mpl.colors.Normalize(vmin = 0, vmax = size, clip = False))
    print ("number of communities: ", int(size), " modularity: ", community.modularity (partition, G))
    for com in set(partition.values()) :
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 100, node_color = cm.to_rgba (count))
        count = count + 1.
        print(list_nodes)
        print(len(list_nodes))
    nx.draw_networkx_edges(G,pos, alpha=0.2)
    plt.show()
    
partition_and_draw (G)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




