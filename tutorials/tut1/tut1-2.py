# %%
# %matplotlib inline
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")		# fix to make sure plots actually display
from matplotlib import pyplot as plt
import pandas as pd
import os

# %%
path_to_networks = "data/"
G = nx.read_edgelist(path_to_networks + "protein.edgelist.txt", delimiter='\t', create_using=nx.Graph())


# %%
# the number of vertices,number  of  edges,  average  degree,  maximum  degree,  diameter,  average  shortest  pathlength, and global clustering coefficient
def get_vertext_count(G):
    unique_vertices = set([node for node, adj in G.adjacency()])
    vertext_count = len(unique_vertices)
    return vertext_count, unique_vertices
    
n_cnt, N = get_vertext_count(G)
print(n_cnt)
# %%
def get_edge_count(G):
    edges = [adj for node, adj in G.adjacency()]
    sum_edges = sum([len(edge) for edge in edges])
    return sum_edges, edges

e_cnt, L = get_edge_count(G)
print(e_cnt)
# %%

def get_average_degree(edge_cnt, node_cnt):
    return edge_cnt/node_cnt # The directed case but data is stored as directed graphs
    
avg_degree = get_average_degree(e_cnt,n_cnt)
print(avg_degree)

# %%
plt.plot([1,2,3,111,3,43,2,1,1,2,2])
plt.show()
# %%


# %%
