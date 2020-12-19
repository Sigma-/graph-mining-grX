#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

data = pd.read_csv('tij_InVS.dat', header = 1)
data.columns = ['tij']
data


# In[2]:


def load_sociopatterns_network():
    df = pd.DataFrame(data.tij.str.split(' ',3).tolist(),columns=['t','i','j'])
    print(df)
    #df['weigth'] = '1'
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


# In[3]:


def drawGraph (G, pos = None):
    plt.figure (figsize=(20,20))
    if not pos:
        pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_labels(G, pos) 
    plt.show()
    
drawGraph(G)


# In[4]:


#nbr nodes & edges 
len(G.nodes()), len(G.edges())


# In[5]:


#degré de centralité
dcs = pd.Series(nx.degree_centrality(G))
dcsasc = dcs.sort_values(ascending=False)
dcsasc


# In[6]:


#centralité 2
dbs = pd.Series(nx.betweenness_centrality(G))
dbs


# In[7]:


#noeud avec le plus de liens
def rank_ordered_neighbors(G):

    s = pd.Series({n: len(list(G.neighbors(n))) for n in G.nodes()})
    return s.sort_values(ascending=False)

rank_ordered_neighbors(G)


# In[8]:


from nxviz import MatrixPlot


# # 2-degree_distribution

# In[ ]:


import collections
# extract the degree sequence
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
#print("Degree sequence", degree_sequence)

# compute the number of nodes having a given centrality
degreeCount = collections.Counter(degree_sequence)
#print("degreeCount", degreeCount)

# spilt into two differents list the degree and the number of nodes having that degree
deg, cnt = zip(*degreeCount.items())
#print(deg, cnt)

# build and draw histogram
plt.bar(deg, cnt, width=0.5, color=(0, 0.4, 0.5, 1))

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d for d in deg])
ax.set_xticklabels(deg)

plt.show()


# # 3-clustering_coefficient

# In[ ]:


# compute and display the average clustering coefficient of the karate club graph
print(nx.average_clustering(G))


# # 4-configuration_model
# 

# In[ ]:


import community
plt.figure(figsize=((10,8))
           
partition = community.best_partition(G[0])
size = float(len(set(partition.values())))
pos = nx.kamada_kawai_layout(G[0])
count = 0
colors = ['red', 'blue', 'yellow', 'black',
          'brown', 'purple', 'green', 'pink']
for com in set(partition.values()):
    list_nodes = [nodes for
                  nodes in partition.keys()
                if partition[nodes] == com]
    nx.draw_networkx_nodes(G[0],
        pos, list_nodes, node_size = 20,
        node_color = colors[count])
    count = count + 1

nx.draw_networkx_edges(G[0], pos, alpha=0.2)

plt.plot(run_before, run_after, 'ro-')
plt.plot(walk_before, walk_after, 'bo-')
plt.show()


# In[29]:


#!pip3 install python-louvain
import community
import matplotlib as mpl

def partition_and_draw (G):
    pos = nx.spring_layout(G)
    partition = community.best_partition(G)
    size = float(len(set(partition.values())))
    count = 0.
    cm = plt.cm.ScalarMappable(cmap = plt.get_cmap('Paired'), 
                                norm = mpl.colors.Normalize(vmin = 0, vmax = size, clip = False))
    print ("number of communities: ", int(size), " modularity: ", community.modularity (partition, G))
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 60, node_color = cm.to_rgba (count))


    nx.draw_networkx_edges(G,pos, alpha=0.5)
    plt.show()
    
partition_and_draw (G)


# In[ ]:




