#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    nx.draw_networkx_labels(G, pos) 
    plt.show()


# In[2]:




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
    nx.draw_networkx_labels(G, pos, alpha=0.1) 
    plt.show()
    

G = load_sociopatterns_network()
drawGraph(G)

#j'ai changé le Alpha 


# In[3]:


#nbr nodes & edges 
len(G.nodes()), len(G.edges())


# In[4]:


#degré de centralité
dcs = pd.Series(nx.degree_centrality(G))
dcsasc = dcs.sort_values(ascending=False)
dcsasc


# In[5]:


#centralité 2
dbs = pd.Series(nx.betweenness_centrality(G))
dbs


# In[6]:


#noeud avec le plus de liens
def rank_ordered_neighbors(G):

    s = pd.Series({n: len(list(G.neighbors(n))) for n in G.nodes()})
    return s.sort_values(ascending=False)

rank_ordered_neighbors(G)


# # 2-degree_distribution
# 

# In[7]:


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
#ax.set_xticks([d for d in deg])
#ax.set_xticklabels(deg)

plt.show()

#j'ai retiré les deux lignes ax 


# # 3-clustering_coefficient

# In[8]:


# compute and display the average clustering coefficient of the karate club graph
print(nx.average_clustering(G))

