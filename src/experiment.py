# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# # %%
# get_ipython().system('conda install --yes --prefix {sys.prefix} matplotlib pandas scikit-learn scipy networkx jupyter')
# # !conda install --yes --prefix {sys.prefix} -c conda-forge/label/cf202003 infomap
# get_ipython().system('{sys.executable} -m pip install python-louvain multiprocess')
# get_ipython().system('{sys.executable} -m pip install -e ../vendor/py')
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
import networkx as nx
from networkx.algorithms import community as algorithms
from networkx.generators import community as generator
from networkx.algorithms.community.quality import modularity
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
import math 
from collections import OrderedDict, Counter, deque
import itertools
import numpy as np
import pandas as pd
# import multiprocessing as mp
import multiprocess as mp
import matplotlib.cm as cm
import community as community_louvain
import scipy
from random import random
import operator
from utils import post_transform, compute_experiment, modularity_wrapper, flake_odf_wrapper, map_equation_wrapper, coverage_wrapper
from algorithms.louvain_core import LouvainCoreAlgorithm

def extract_true_communities(G):
    communities = []
    for v in G:
        curr_community = G.nodes[v]['community']
        if not any([v in community for community in communities]):
            communities.append(frozenset(curr_community))
    partition = extract_partition_map(communities)
    return partition, communities

def extract_partition_map(communities):
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
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                            cmap=cmap, node_color=list(partition.values()), ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=40, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    return None

def sort_partition_map(partition_map):
    return dict(sorted(partition_map.items()))    

# def post_transform(algorithm):
#     def modified_function(G):
#         communities = algorithm(G)
#         pred_partition_map = extract_partition_map(communities)
#         return pred_partition_map
#     modified_function.__name__ = algorithm.__name__
#     return modified_function

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
    joint_entorpy = -np.nansum(prob_matrix*np.log2(prob_matrix))
    return joint_entorpy

def normalized_mutual_information(true_partitions, pred_partitions):
    H_X = compute_entropy(true_partitions)
    H_Y = compute_entropy(pred_partitions)
    H_XY = compute_joint_entropy(true_partitions, pred_partitions)
    conditional_entropy =  compute_conditional_entropy(H_X, H_Y, H_XY)
    sum_of_independent_entropies = H_X + H_Y
    nominator = 2 * conditional_entropy
    denominator = sum_of_independent_entropies
    return nominator/denominator

def run_experiment(algorithms_for_experiment, backup):

    collected_data = []
    iterations = list(range(0, 10))
    node_sizes = [250]
    mus = np.arange(0.1, 0.8, 0.1)
    configuration_set = itertools.product(*[iterations, algorithms_for_experiment.items(), node_sizes, mus])


    cpu_cnt = 8
    print(cpu_cnt)
    # with mp.Pool(cpu_cnt) as pool:
    pool = mp.Pool(processes=cpu_cnt)
    print(f"Running experiments in parallel with {cpu_cnt} cpus")
    parallel_execution_data = pool.imap_unordered(compute_experiment, configuration_set)
    
    for result in parallel_execution_data:
        save_data(result, backup)
        save_data(result, result["method"])

def save_data(result, filename):
    with open(f"{filename.lower().replace(' ', '_')}.csv", 'a') as f:
        print("Writing into file")
        print(result)
        f.write(", ".join(map(str, result.values()))+"\n")



def draw_plots(data):
    aggregated_over_trials = data.groupby(['method', 'N', 'µ']).mean()
    grouped_by_algorithm = aggregated_over_trials.groupby(['method'])
    num_groups = len(grouped_by_algorithm)
    num_rows = int(math.ceil(num_groups/2))
    tmp = None
    fig, axes = plt.subplots(num_rows, 2, sharex=True, sharey=True) 
    fig.set_size_inches(10, 5*num_rows)
    axes = axes.flatten()
    for idx, (algorithm_name, algorithm_data) in enumerate(grouped_by_algorithm):
        axes[idx].set_title(algorithm_name)
        tmp = algorithm_data.reset_index().pivot(index='µ', columns='N', values='NMI')
        tmp.plot(ax=axes[idx])
        axes[idx].set_ylabel("Normalized Mutual Information")

    if num_groups < len(axes):
        fig.delaxes(*axes[num_groups:])
    fig.set_tight_layout(True)
    return plt.show()

if __name__ == "__main__":
    louvain_algorithm = LouvainCoreAlgorithm(fitness_function=modularity_wrapper, verbose=True, max_iter=20)
    infomap_algorithm = LouvainCoreAlgorithm(fitness_function=map_equation_wrapper, verbose=True, max_iter=20)
    coverage_algorithm = LouvainCoreAlgorithm(fitness_function=coverage_wrapper, verbose=True, max_iter=20)
    flake_algorithm = LouvainCoreAlgorithm(fitness_function=flake_odf_wrapper, verbose=True, max_iter=20)
    lblprob_algorithm = post_transform(algorithms.asyn_lpa_communities)

    algorithms_for_experiment = {
        # infomap_algorithm.run:"Map Equation",
        # coverage_algorithm.run:"Coverage Maximization",
        flake_algorithm.run:"Flake Maximization",
        # lblprob_algorithm:"Labelpropagation Algorithm",
        # louvain_algorithm.run: "Louvain Algorithm",
    }

    run_experiment(algorithms_for_experiment, "backup")




