import matplotlib.pyplot as plt
import networkx as nx
import networkx.generators.community as generator
from networkx.algorithms import community as algorithms
from collections import OrderedDict
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
import math 
import itertools
from collections import OrderedDict, Counter, defaultdict
import pandas as pd
from algorithms.map_equation import map_equation
import community as community_louvain
import io
import matplotlib.cm as cm



def extract_true_communities(G):
    communities = []
    for v in G:
        curr_community = G.nodes[v]['community']
        if not any([v in community for community in communities]):
            communities.append(frozenset(curr_community))
    partition = extract_partition_map(communities)
    return partition, communities

def extract_partition_map(communities):
    # print(communities)
    node_community_participation = {node:idx for idx, community in enumerate(communities) for node in community}
    return OrderedDict(sorted(node_community_participation.items()))

def extract_community_map(partition):
    v = {}
    for key, value in partition.items():
        v.setdefault(value, []).append(key)
    communities = list(dict(sorted(v.items())).values())
    return communities

def generate_benchmark_graph(n, mu=0.1):
    tau1 = 2 # Degree distribution power-law exponent
    tau2 = 1.1 # Community size distribtution power-law exponent
    G = generator.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=20, max_degree=50, max_iters=1000, seed=10)
    pos = nx.spring_layout(G, k=.3)
    return G, pos

def visualize_benchmark_graph(G, pos, partition = None, ax=None):
    if partition:
        prt2idx = {key: idx for idx, key in enumerate(set(partition.values()))}
        partition = {node:prt2idx[prt] for node, prt in partition.items()}
        cmap = cm.get_cmap('jet', max(set(partition.values())) + 1)
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                            cmap=cmap, node_color=list(partition.values()), ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=40, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    return None 

def sort_partition_map(partition_map):
    return dict(sorted(partition_map.items())) 

def compute_experiment(configuration):
    iteration, predictor, node_size, mu = configuration
    print(f"{iteration}. iteration for config {(node_size, mu)} with algorithm {predictor[1]}")
    G, pos = generate_benchmark_graph(node_size,mu)
    true_partition_map, communities = extract_true_communities(G)
    pred_partition_map = predictor[0](G)
    # pred_partition_map = extract_partition_map(communities)
    nmi = normalized_mutual_information(true_partition_map, pred_partition_map)
    result = {"method":predictor[1], "N":node_size, "µ":mu, "NMI":nmi}
    # io.open("intermediate_results.csv")
    return result

def post_transform(algorithm):
    def modified_function(G):
        communities = algorithm(G)
        pred_partition_map = extract_partition_map(communities)
        return pred_partition_map
    modified_function.__name__ = algorithm.__name__
    return modified_function

def modified_girvan_newman():
    def modified_function(G):
        communities = next(algorithms.girvan_newman(G))
        pred_partition_map = extract_partition_map(communities)
        return pred_partition_map
    return modified_function

def compute_entropy(partition_map):
    class_counts = np.array(list(Counter(partition_map.values()).values()))
    class_probabilities = class_counts/sum(class_counts) 
    partial_entropies = - class_probabilities * np.log2(class_probabilities)
    entropy = partial_entropies.sum() 
    return entropy

def compute_conditional_entropy(hx, hy, hxy):
    return hx + hy - hxy

def compute_joint_entropy(true_partitions, pred_partitions):
    cnt_matrix = contingency_matrix(list(true_partitions.values()), list(pred_partitions.values())).T
    matrix_sum = cnt_matrix.sum()
    prob_matrix = cnt_matrix / matrix_sum
    joint_entropy = -np.nansum(prob_matrix*np.log2(prob_matrix))
    return joint_entropy

def normalized_mutual_information(true_partitions, pred_partitions):
    H_X = compute_entropy(true_partitions)
    H_Y = compute_entropy(pred_partitions)
    H_XY = compute_joint_entropy(true_partitions, pred_partitions)
    conditional_entropy =  compute_conditional_entropy(H_X, H_Y, H_XY)
    sum_of_independent_entropies = H_X + H_Y
    nominator = 2 * conditional_entropy
    denominator = sum_of_independent_entropies
    return nominator/denominator

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
    # print(partitions)
    for node, community in partition_map.items():
        node_indegrees[node] = sum(True for adj_node in G[node] if partition_map[adj_node] == community)
        node_exdegrees[node] = len(G[node])/2
        node_diff[node] = node_indegrees[node] < node_exdegrees[node]
        partition_indegrees[community] += int(node_diff[node])
        partition_degrees[community] += 1

    fraction = np.array(list(partition_indegrees.values()))/np.array(list(partition_degrees.values()))
    return -fraction.mean()