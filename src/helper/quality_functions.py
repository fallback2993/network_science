from collections import defaultdict

import community as community_louvain
import numpy as np
from networkx.algorithms import community as algorithms

from algorithms.map_equation import map_equation
from helper.utils import extract_community_map


def map_equation_wrapper(partition, G):
    L, index_codelength, module_codelength = map_equation(G, partition)
    return -L


def modularity_wrapper(partition, G):
    return community_louvain.modularity(partition, G)


def coverage_wrapper(partition, G):
    community_map = extract_community_map(partition)
    return algorithms.coverage(G, community_map)


def flake_odf_wrapper(partition_map, G):
    partitions = np.unique(list(partition_map.values()))
    if len(partitions) < 2:
        return -1
    partition_indegrees = defaultdict(float)
    partition_degrees = defaultdict(float)
    node_indegrees = {}
    node_exdegrees = {}
    node_diff = {}
    for node, community in partition_map.items():
        node_indegrees[node] = sum(True for adj_node in G[node] if partition_map[adj_node] == community)
        node_exdegrees[node] = len(G[node]) / 2
        node_diff[node] = node_indegrees[node] < node_exdegrees[node]
        partition_indegrees[community] += int(node_diff[node])
        partition_degrees[community] += 1

    fraction = np.array(list(partition_indegrees.values())) / np.array(list(partition_degrees.values()))
    return -fraction.mean()


def conductance_wrapper(partition_map, G):
    # partitions = np.unique(list(partition_map.values()))
    partition_edges = defaultdict(float)
    partition_nodes = defaultdict(float)
    node_edges = {}
    for node, community in partition_map.items():
        node_edges[node] = sum(True for adj_node in G[node] if partition_map[adj_node] != community)
        partition_nodes[community] += int(node_edges[node])
        partition_edges[community] += int(len(G[node])) - int(node_edges[node])

    cs = np.array(list(partition_nodes.values()))
    ms = np.array(list(partition_edges.values()))

    fraction = cs / ((ms) + cs)

    return -fraction.mean()


def random_wrapper(partition_map, G):
    return np.random.uniform()
