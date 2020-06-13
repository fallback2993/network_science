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
    if len(G.nodes()) < 2:
        return 1.0, 0, 1.0

    num_links = len(G.edges())
    num_nodes = len(G.nodes())
    unique_partitions = np.unique(list(partition_map.values()))
    num_partitions = len(unique_partitions)

    node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
    id2node = dict(enumerate(node2id.keys()))
    comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
    id2comm = dict(enumerate(comm2id.keys()))
    original_community_map = dict(enumerate(extract_community_map(partition_map)))  # For some reason partition zero misses
    community_map = {idx: [node2id[node] for node in community] for idx, community in original_community_map.items()}
    partition_map = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
    G = nx.relabel_nodes(G, node2id)
    A = np.array(nx.adjacency_matrix(G).todense())

    num_links = len(G.edges())
    num_nodes = len(G.nodes())

    # adjacent_partition_matrix = np.full([num_nodes, num_nodes], np.nan)
    # for name, community in community_map.items():
    #     for node in community:
    #         old_node_name = id2node[node]
    #         adjacent_partition_matrix[node][node] = partition_map[node]
    #         for adjacent_node in G[old_node_name]:
    #             adjacent_node = node2id[adjacent_node]
    #             adjacent_partition_matrix[node][adjacent_node] = partition_map[adjacent_node]

    # print(A)
    # print(adjacent_partition_matrix)
    # G = nx.relabel_nodes(G, node2id)
    # A = np.array(nx.adjacency_matrix(G).todense())

    # num_links = len(G.edges())
    # num_nodes = len(G.nodes())

    adjacent_partition_matrix = np.full([num_nodes, num_nodes], np.nan)
    for node in G.nodes():
        adjacent_partition_matrix[node][node] = partition_map[node]
        for adj_node in G.nodes():
            if np.all(A[node][adj_node]):
                adjacent_partition_matrix[node][adj_node] = partition_map[adj_node]

    diagonal = np.diagonal(adjacent_partition_matrix)

    tmp = adjacent_partition_matrix.copy()
    np.fill_diagonal(tmp, np.nan)
    zm2 = np.ma.masked_where(np.isnan(adjacent_partition_matrix), A)
    zm = np.ma.masked_where(np.isnan(adjacent_partition_matrix), adjacent_partition_matrix)
    apm_linkage = zm == diagonal[:, None]
    A_in = np.ma.masked_where(np.invert(apm_linkage), A)
    A_ex = np.ma.masked_where(apm_linkage == True, A)
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
    partition_links = (partition_in_links + partition_ex_links)
    partition_exit_prob = np.nan_to_num(partition_ex_links / partition_links) / num_partitions

    node_weights = np.array(A.sum(axis=0)).squeeze()
    p_u = node_weights / node_weights.sum()

    for name, community in community_map.items():
        partition_probabilities[name] = sum(p_u[community])

    p_a_i = partition_probabilities
    q_out_i = partition_exit_prob
    q_out = q_out_i.sum()
    p_circle_i = p_a_i + q_out_i

    print(p_a_i, "\n", q_out_i, "\n", q_out, "\n", p_circle_i, "\n", p_u, "\n")

    H_Q = -sum(np.nan_to_num((q_out_i / q_out) * np.log2(q_out_i / q_out)))
    term1 = -np.nan_to_num((q_out_i / p_circle_i) * np.log2((q_out_i / p_circle_i)))
    term2 = np.zeros(num_partitions)
    for name, community in community_map.items():
        term2[name] = -compute_weighted_entropy(p_u[community], p_circle_i[name])

    H_P_i = term1 + term2
    print(community_map)
    print(H_P_i, "\n", H_Q, "\n", q_out, "\n", p_circle_i, "\n", term1, "\n", term2, "\n", p_u, "\n")

    index_codelength = q_out * H_Q
    module_codelength = p_circle_i.dot(H_P_i)
    L = index_codelength + module_codelength
    L = np.asarray(L).flatten()[0]

    return L, index_codelength, module_codelength


def map_equation_improved(G, partition_map, DEBUG=False):
    if len(G.nodes()) < 2:
        return 1.0, 0, 1.0

    unique_partitions = np.unique(list(partition_map.values()))
    num_partitions = len(unique_partitions)

    node2id = dict({node: idx for idx, node in enumerate(G)})
    id2node = dict(enumerate(node2id.keys()))
    comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
    partition_map = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
    original_community_map = dict(enumerate(extract_community_map(partition_map)))  # For some reason partition zero misses
    community_map = {idx: [node2id[node] for node in community] for idx, community in original_community_map.items()}
    G = nx.relabel_nodes(G, node2id)

    node_partition_in_links, node_partition_ex_links, node_weights, node_partitions = map_equation_essentials(G, partition_map, DEBUG)
    p_a_i, q_out_i, q_out, p_circle_i, p_u = retrieve_linkings(node_partition_in_links, node_partition_ex_links, node_weights, node_partitions)
    L, index_codelength, module_codelength = compute_minimal_codelength(p_a_i, q_out_i, q_out, p_circle_i, p_u, node_partitions)
    return L, index_codelength, module_codelength


def extract_community_map(partition):
    v = {}
    for key, value in partition.items():
        v.setdefault(value, []).append(key)
    communities = list(dict(sorted(v.items())).values())
    return communities


def compute_weighted_entropy(probs, normalizer):
    return np.nansum((probs / normalizer) * np.log2(probs / normalizer))


def map_equation_essentials(G, partition_map, DEBUG=False):
    print("")
    # node2id = dict({node: idx for idx, node in enumerate(set(G))})

    A = np.array(nx.adjacency_matrix(G).todense())
    num_nodes = len(G.nodes())

    adjacent_partition_matrix = np.full([num_nodes, num_nodes], np.nan)
    for node in G.nodes():
        adjacent_partition_matrix[node][node] = partition_map[node]
        for adj_node in G.nodes():
            if np.all(A[node][adj_node]):
                adjacent_partition_matrix[node][adj_node] = partition_map[adj_node]

    # print(A)
    # print(adjacent_partition_matrix)

    diagonal = np.diagonal(adjacent_partition_matrix)

    # tmp = adjacent_partition_matrix.copy()
    # np.fill_diagonal(tmp, np.nan)
    zm = np.ma.masked_where(np.isnan(adjacent_partition_matrix), adjacent_partition_matrix)
    apm_linkage = zm == diagonal[:, None]
    A_in = np.ma.masked_where(np.invert(apm_linkage), A)
    A_ex = np.ma.masked_where(apm_linkage, A)  # Where is True
    # partitions = np.array(list(partition_map.values()))
    # unique_partitions = np.unique(partitions)

    # num_partitions = len(unique_partitions)

    node_partition_in_links = np.array(A_in.sum(axis=1)).flatten()
    node_partition_ex_links = np.array(A_ex.sum(axis=1)).flatten()
    node_weights = np.array(A.sum(axis=0)).squeeze()

    return node_partition_in_links, node_partition_ex_links, node_weights, diagonal


def retrieve_linkings(node_partition_in_links, node_partition_ex_links, node_weights, partition_mapping):
    unique_partitions = np.unique(partition_mapping)
    num_partitions = len(unique_partitions)
    partition_ex_links = np.zeros(num_partitions)
    partition_in_links = np.zeros(num_partitions)
    for partition in unique_partitions:
        partition = int(partition)
        indices_to_check = list(np.where(partition_mapping == partition)[0])
        partition_ex_links[partition] = sum(node_partition_ex_links[indices_to_check])
        partition_in_links[partition] = sum(node_partition_in_links[indices_to_check])

    # print(partition_ex_links)
    # print(partition_in_links)
    # if DEBUG:
    #     adjacent_nodes_per_node = {
    #         node: (partition_map[node], [partition_map[adjacent_node] for adjacent_node in G[node]])
    #         for name, community in community_map.items() for node in community
    #     }
    #     adjacent_nodes_per_node = dict(sorted(adjacent_nodes_per_node.items()))
    #     for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items():
    #         partition_ex_links[node_partition] += sum(node_partition != np.array(neighbor_partitions))
    #         partition_in_links[node_partition] += sum(node_partition == np.array(neighbor_partitions))

    partition_probabilities = np.zeros(num_partitions)
    partition_links = (partition_in_links + partition_ex_links)
    partition_exit_prob = np.nan_to_num(partition_ex_links / partition_links) / num_partitions

    p_u = node_weights / node_weights.sum()

    for node_weight, community in zip(p_u, partition_mapping):
        partition_probabilities[int(community)] += node_weight

    p_a_i = partition_probabilities
    q_out_i = partition_exit_prob
    q_out = q_out_i.sum()
    p_circle_i = p_a_i + q_out_i

    print(p_a_i, "\n", q_out_i, "\n", q_out, "\n", p_circle_i, "\n", p_u, "\n")
    return p_a_i, q_out_i, q_out, p_circle_i, p_u


def compute_minimal_codelength(p_a_i, q_out_i, q_out, p_circle_i, p_u, partition_mapping):

    H_Q = -sum(np.nan_to_num((q_out_i / q_out) * np.log2(q_out_i / q_out)))
    term1 = -np.nan_to_num((q_out_i / p_circle_i) * np.log2((q_out_i / p_circle_i)))
    term2 = np.zeros(len(np.unique(partition_mapping)))

    for node_weight, community in zip(p_u, partition_mapping):
        community = int(community)
        term2[community] += node_weight/p_circle_i[community] * np.log2(node_weight/p_circle_i[community]) if node_weight != 0 else 0
    # for name, community in community_map.items():
    #     term2[name] = -compute_weighted_entropy(p_u[community], p_circle_i[name])

    H_P_i = term1 - term2

    print(partition_mapping)
    print(H_P_i, "\n", H_Q, "\n", q_out, "\n", p_circle_i, "\n", term1, "\n", term2, "\n", p_u, "\n")

    index_codelength = q_out * H_Q
    module_codelength = p_circle_i.dot(H_P_i)
    L = index_codelength + module_codelength
    L = np.asarray(L).flatten()[0]
    return L, index_codelength, module_codelength
