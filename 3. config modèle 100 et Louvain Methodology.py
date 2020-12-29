#!/usr/bin/env python
# coding: utf-8

# In[19]:


# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:03:36 2020

@author: Nano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import random 
import operator
import collections
import pydot
import community
import matplotlib as mpl


data = pd.read_csv('tij_InVS.dat', header = 1)
data.columns = ['tij']

def load_sociopatterns_network():
    df = pd.DataFrame(data.tij.str.split(' ',3).tolist(),columns=['t','i','j'])
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

modularities = []
for i in range(1,101):
    degree_sequence = ([d for n, d in G.degree()])
    modelConfig=nx.configuration_model(degree_sequence)
    model_degree_sequence = ([d for n, d in modelConfig.degree()])


    modelConfig=nx.Graph(modelConfig) #suppression des liens multiples
    model_degree_without_mul = ([d for n, d in modelConfig.degree()])


    modelConfig.remove_edges_from(nx.selfloop_edges(modelConfig)) # suppression des boucles
    numberOfInitialEdges= G.number_of_edges()
    numberOfFinalEdges = modelConfig.number_of_edges()

    pos = nx.spring_layout(G)
    partition = community.best_partition(G)
    size = float(len(set(partition.values())))
    count = 0.
    #cmap = plt.get_cmap("tab10")
    cm = plt.cm.ScalarMappable(cmap = plt.get_cmap('Paired'), 
    norm = mpl.colors.Normalize(vmin = 0, vmax = size, clip = False))
    #print ("number of communities: ", int(size)+1, " modularity: ", community.modularity (partition, G))
    modularities.append(community.modularity (partition, G))
print(len(modularities))


# In[46]:


def draw_modularity(modularities):
    plt.subplot(111)
    
    plt.plot([i for i in range(1,len(modularities)+1)], list(modularities))
    
    plt.xlabel('Graphe', fontsize=15)
    plt.ylabel('Modularité', fontsize=15)
    plt.title("Modularité par graphe", fontsize=15)
    plt.ylim(0.4, 0.8)

    plt.rcParams["figure.figsize"] = [12, 10]
    plt.show()

draw_modularity(modularities)


# In[47]:


def moyenne(tableau):
    return sum(tableau, 0.0) / len(tableau)
def variance(tableau):
    m=moyenne(tableau)
    return moyenne([(x-m)**2 for x in tableau])
def ecartype(tableau):
    return variance(tableau)**0.5

print(moyenne(modularities))


# In[ ]:





# In[ ]:





# In[ ]:




