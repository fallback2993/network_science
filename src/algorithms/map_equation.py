import networkx as nx
# from networkx.algorithms import community as algorithms
# from networkx.generators import community as generator
# from networkx.algorithms.community.quality import modularity
# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics.cluster import contingency_matrix
# from sklearn.metrics.cluster import normalized_mutual_info_score
import math 
import itertools
from collections import OrderedDict, Counter, deque
# import pandas as pd
# import multiprocess as mp
# import matplotlib.cm as cm
# import community as community_louvain
# import scipy
# from random import random
# import operator

def map_equation(G, partition_map):
    def compute_weighted_entropy(probs, normalizer):
        return np.nansum((probs/normalizer) * np.log2(probs/normalizer))
    if len(G.nodes()) < 2:
        return 1.0, 0, 1.0
    
    num_links = len(G.edges())
    num_nodes = len(G.nodes())
    unique_partitions = np.unique(list(partition_map.values()))
    num_partitions = len(unique_partitions)

    p_a_i, q_out_i, q_out, p_circle_i,p_u, partition_map, community_map = map_equation_essentials(G, partition_map)
    H_Q = - sum(np.nan_to_num((q_out_i/q_out) * np.log2(q_out_i/q_out)))
    term1 = -np.nan_to_num((q_out_i/p_circle_i)*np.log2((q_out_i/p_circle_i)))
    term2 = np.zeros(num_partitions)
    for name, community in community_map.items():
        term2[name] = -compute_weighted_entropy(p_u[community], p_circle_i[name])
    
    H_P_i = term1 + term2
    index_codelength = q_out * H_Q 
    module_codelength = p_circle_i.dot(H_P_i)
    L = index_codelength + module_codelength 
    L = np.asarray(L).flatten()[0]

    return L, index_codelength, module_codelength

def extract_community_map(partition):
    v = {}
    for key, value in partition.items():
        v.setdefault(value, []).append(key)
    communities = list(dict(sorted(v.items())).values())
    return communities

def map_equation_essentials(G, partition_map):
    A = nx.adjacency_matrix(G).todense()
    
    node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
    id2node = dict(enumerate(node2id.keys()))
    comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
    id2comm = dict(enumerate(comm2id.keys()))
    original_community_map = dict(enumerate(extract_community_map(partition_map))) # For some reason partition zero misses
    community_map = {idx:[node2id[node] for node in community] for idx, community in original_community_map.items()}
    partition_map = {node2id[node]:comm2id[community] for node, community in partition_map.items()}

    num_links = len(G.edges())
    num_nodes = len(G.nodes())

    adjacent_partition_matrix = np.full([num_nodes, num_nodes], np.nan)
    for name, community in community_map.items(): 
        for node in community:
            old_node_name = id2node[node]
            adjacent_partition_matrix[node][node] = partition_map[node]
            for adjacent_node in G[old_node_name]:
                adjacent_node = node2id[adjacent_node]
                adjacent_partition_matrix[node][adjacent_node] = partition_map[adjacent_node]
    diagonal = np.diagonal(adjacent_partition_matrix)

    tmp = adjacent_partition_matrix.copy()
    np.fill_diagonal(tmp, np.nan)
    zm2 = np.ma.masked_where(np.isnan(adjacent_partition_matrix), A)
    zm = np.ma.masked_where(np.isnan(adjacent_partition_matrix), adjacent_partition_matrix)
    apm_linkage = zm == diagonal[:, None]
    A_in = np.ma.masked_where(np.invert(apm_linkage), A)
    A_ex = np.ma.masked_where(apm_linkage==True, A)
    unique_partitions = np.unique(list(partition_map.values()))

    num_partitions = len(unique_partitions)
    partition_ex_links = np.zeros(num_partitions)
    partition_in_links = np.zeros(num_partitions)
        
    node_partition_in_links = np.array(A_in.sum(axis=1)).flatten()
    node_partition_ex_links = np.array(A_ex.sum(axis=1)).flatten()
        
    for partition in unique_partitions:
        partition = int(partition)
        indices_to_check = list(np.where(diagonal == partition)[0])
        partition_in_links[partition] = sum(node_partition_in_links[indices_to_check])
        partition_ex_links[partition] = sum(node_partition_ex_links[indices_to_check])

    partition_probabilities = np.zeros(num_partitions)
    partition_links = (partition_in_links+partition_ex_links)
    partition_exit_prob = np.nan_to_num(partition_ex_links/partition_links)/num_partitions

    node_weights = np.array(A.sum(axis=0)).squeeze()
    p_u = node_weights / node_weights.sum()

    for name, community in community_map.items():
        partition_probabilities[name] = sum(p_u[community])

    p_a_i = partition_probabilities
    q_out_i = partition_exit_prob
    q_out = q_out_i.sum()
    p_circle_i = p_a_i + q_out_i

    return p_a_i, q_out_i, q_out, p_circle_i, p_u, partition_map, community_map