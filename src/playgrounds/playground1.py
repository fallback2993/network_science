#%%
import sys
!conda install --yes --prefix {sys.prefix} matplotlib pandas scikit-learn scipy networkx jupyter
# !conda install --yes --prefix {sys.prefix} -c conda-forge/label/cf202003 infomap
!{sys.executable} -m pip install python-louvain multiprocess pquality 
!{sys.executable} -m pip install -e ../vendor/py
%matplotlib inline
# %%
import networkx as nx
from networkx.algorithms import community as algorithms
from networkx.generators import community as generator
from networkx.algorithms.community.quality import modularity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
import math 
import itertools
from collections import OrderedDict, Counter, deque, defaultdict
import pandas as pd
import multiprocess as mp
import matplotlib.cm as cm
import community as community_louvain
import scipy
from random import random
import operator
import time


# %% [markdown]
# # This notebook show cases some initial introductory code examples with networkx

# %% [markdown]
# ## Example drawing of a simple community graph
# %%
G = nx.barbell_graph(5, 1)
pos = nx.spring_layout(G)
communities_generator = algorithms.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
partition = {node_name:idx+1 for idx, community in enumerate(next_level_communities) for node_name in community}
nx.draw_networkx(G, pos, edgecolors="black", node_size=600, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))


# next_level_communities

# %% [markdown]
# ## Example drawing of a LFR community graph


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
    # return nx.draw_networkx(G, 
    # pos, 
    # edge_color="black", 
    # with_labels=False, 
    # node_size=50, 
    # cmap=plt.cm.viridis, 
    # node_color=list(partition.values()) if partition else None, 
    # ax=ax)

# %%
print(f"Computing the ground truth of the LFR graph")
G, pos = generate_benchmark_graph(250,0.1)
true_partition_map, communities = extract_true_communities(G)
communities = algorithms.greedy_modularity_communities(G)
cnm_partition_map = extract_partition_map(communities)
communities = algorithms.asyn_lpa_communities(G)
lpa_partition_map = extract_partition_map(communities)
louvain_partition_map = community_louvain.best_partition(G)

#%%
print(f"Drawing the the LFR graphs")
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(10, 10)
ax[0][0].set_title(f"Initial Graph", fontsize=10)
ax[0][0].set_axis_off()
ax[0][1].set_title(f"Ground-Truth", fontsize=10)
ax[0][1].set_axis_off()
# ax[0][1].set_title(f"Louvain", fontsize=10)
# ax[0][1].set_axis_off()
ax[1][0].set_title(f"Greedy Max Modularity ", fontsize=10)
ax[1][0].set_axis_off()
ax[1][1].set_title(f"Label-Propagation", fontsize=10)
ax[1][1].set_axis_off()
visualize_benchmark_graph(G, pos, None, ax[0][0])
visualize_benchmark_graph(G, pos, true_partition_map, ax[0][1])
# visualize_benchmark_graph(G, pos, louvain_partition_map, ax[0][1])
visualize_benchmark_graph(G, pos, cnm_partition_map, ax[1][0])
visualize_benchmark_graph(G, pos, lpa_partition_map, ax[1][1])
# fig.delaxes(ax[0][1])
plt.tight_layout()
plt.show()
# plt.show()

 # %% [markdown]
# ## Example computing normalized mutual information with all the partitions


# %%
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

true_nmi = normalized_mutual_information(true_partition_map, true_partition_map)
cnm_nmi = normalized_mutual_information(true_partition_map, cnm_partition_map)
cnm_nmi_reversed = normalized_mutual_information(cnm_partition_map, true_partition_map)
lpa_nmi = normalized_mutual_information(true_partition_map, lpa_partition_map)

assert math.isclose(true_nmi, 1), f"True nmi does not equal one with the score of {true_nmi}"
print(f"NMI of true prediction with true prediction amounts to {true_nmi} == 1.0.")

# print(f"Sklearn {normalized_mutual_info_score(list(true_partition_map.values()), list(true_partition_map.values()))} == 1")
assert cnm_nmi < 1, f"Clauset-Moore-Newman not below 1 with {cnm_nmi}"
print(f"NMI of ground truth with cnm prediction amounts to {cnm_nmi} < 1")

assert cnm_nmi == cnm_nmi_reversed, f"NMI score for NMI(X,Y) is not the same as NMI(Y,X)"
print(f"Reversed NMI score is {cnm_nmi_reversed}")

assert cnm_nmi < 1, f"Label propagation is not below 1 with {lpa_nmi}"
print(f"NMI of ground truth with lpa prediction amounts to {lpa_nmi} < 1")
# print(f"Sklearn {normalized_mutual_info_score(list(true_partition_map.values()), list(lpa_partition_map.values()))} < 1")


# %% [markdown]
# ## Processing the experiment sequentially

#%%
def compute_experiment(configuration):
    iteration, predictor, node_size, mu = configuration
    print(f"{iteration}. iteration for config {(node_size, mu)} with algorithm {predictor[1]}")
    G, pos = generate_benchmark_graph(node_size,mu)
    true_partition_map, communities = extract_true_communities(G)
    pred_partition_map = predictor[0](G)
    # pred_partition_map = extract_partition_map(communities)
    nmi = normalized_mutual_information(true_partition_map, pred_partition_map)
    return {"method":predictor[1], "N":node_size, "µ":mu, "NMI":nmi}

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

algorithms_for_experiment = {
    # modified_girvan_newman(): "Girvan-Newman",
    post_transform(algorithms.greedy_modularity_communities):"Greedy Modularity Maximization",
    post_transform(algorithms.asyn_lpa_communities):"Label Propagation",
    community_louvain.best_partition:"Louvain Algorithm"
    }
collected_data = []
iterations = list(range(0,1))
node_sizes = [250]
mus = np.arange(0.1, 1.1-0.9, 0.1)
configuration_set = itertools.product(*[iterations, algorithms_for_experiment.items(), node_sizes, mus])

for configuration in configuration_set:
    collected_data.append(compute_experiment(configuration))

data = pd.DataFrame(collected_data)


# %%
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

draw_plots(data)

# %%
G, pos = generate_benchmark_graph(250,0.1)
louvain_partition_map = community_louvain.best_partition(G)

# %%
def modularity_wrapper(partition, G):
    return community_louvain.modularity(partition, G)
    
modularity_wrapper(louvain_partition_map, G)

# %%
def coverage_wrapper(partition, G):
    community_map = extract_community_map(partition)
    return algorithms.coverage(G, community_map)

coverage_wrapper(louvain_partition_map, G)

# %%
def performance_wrapper(partition, G):
    community_map = extract_community_map(partition)
    return algorithms.performance(G, community_map)

performance_wrapper(louvain_partition_map, G)
# %%
# %%time
def map_equation(G, partition_map):
    def compute_weighted_entropy(probs, normalizer):
        return np.nansum((probs/normalizer) * np.log2(probs/normalizer))
    if len(G.nodes()) < 2:
        return 1.0, 0, 1.0
    
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
    node_relative_weight = node_weights / node_weights.sum()

    for name, community in community_map.items():
        partition_probabilities[name] = sum(node_relative_weight[community])
        
    p_a_i = partition_probabilities
    q_out_i = partition_exit_prob
    q_out = q_out_i.sum()
    H_Q = - sum(np.nan_to_num((q_out_i/q_out) * np.log2(q_out_i/q_out)))
    p_circle_i = p_a_i + q_out_i
    term1 = -np.nan_to_num((q_out_i/p_circle_i)*np.log2((q_out_i/p_circle_i)))
    term2 = np.zeros(num_partitions)
    for name, community in community_map.items():
        term2[name] = -compute_weighted_entropy(node_relative_weight[community], p_circle_i[name])
    
    H_P_i = term1 + term2
    index_codelength = q_out * H_Q 
    module_codelength = p_circle_i.dot(H_P_i)
    L = index_codelength + module_codelength 
    L = np.asarray(L).flatten()[0]

    return L, index_codelength, module_codelength

def map_equation_wrapper(partition, G):
    L, index_codelength, module_codelength = map_equation(G, partition) 
    return -L

# louvain_algorithm = LouvainCoreAlgorithm(
#     fitness_function=map_equation_wrapper, verbose=True, max_iter=20, )
# # tmp_G = generator.planted_partition_graph(4,20, p_in=0.9, p_out=0.1)
# tmp_G = nx.karate_club_graph()
# # tmp_G, pos = generate_benchmark_graph(250,0.1)

# init_G, partition_map  = louvain_algorithm.initialize(G=tmp_G)
# # partition_map = dict(enumerate(tmp_G.nodes()))
# bound = 6
# pos = nx.spring_layout(tmp_G)
# fig, ax = plt.subplots(bound, 2)
# fig.set_size_inches(10, 10)
# # VERBOSE = True
# ax[0][0].set_title(map_equation_wrapper(partition_map, tmp_G), fontsize=10)
# ax[0][0].set_axis_off()
# visualize_benchmark_graph(tmp_G, pos, partition_map, ax=ax[0][0])

# for idx in range(1,bound):
#     partition_map, fitness = louvain_algorithm.local_movement(tmp_G, partition_map)
#     louvain_algorithm.levels.append(partition_map)
#     tmp_G, partition_map = louvain_algorithm.reduce_network(tmp_G, partition_map)
#     ax[idx][0].set_title(map_equation_wrapper(partition_map, tmp_G), fontsize=10)
#     ax[idx][0].set_axis_off()
#     visualize_benchmark_graph(tmp_G, pos, partition_map, ax=ax[idx][1])

#     backtracked_map = louvain_algorithm.decode_partition_map(len(louvain_algorithm.levels)-1)
#     ax[idx][1].set_title(map_equation_wrapper(backtracked_map, init_G), fontsize=10)
#     ax[idx][1].set_axis_off()
#     visualize_benchmark_graph(init_G, pos, backtracked_map, ax=ax[idx][0])


# ax[0][1].set_axis_off()
# visualize_benchmark_graph(tmp_G, pos, partition_map, ax=ax[0][1])

# partition_map, fitness = louvain_algorithm.local_movement(tmp_G, partition_map)
# tmp_G, partition_map = louvain_algorithm.reduce_network(tmp_G, partition_map)
# partition_map, fitness = louvain_algorithm.local_movement(tmp_G, partition_map)
# tmp_G, partition_map = louvain_algorithm.reduce_network(tmp_G, partition_map)
# VERBOSE = True
# visualize_benchmark_graph(tmp_G, pos, partition_map, ax=ax[0][1])

# print(map_equation_wrapper(partition_map, tmp_G))

# %%
from infomap import Infomap, MapEquation, InfoNode
from collections import defaultdict 
# g = convert_graph_formats(g_original, nx.Graph)

# G = nx.karate_club_graph()
# G = nx.barbell_graph(5, 1)
# G = nx.bull_graph()
# G = nx.generators.erdos_renyi_graph(10, 0.5)
# G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(5,50, p_in=0.3, p_out=0.01)
G, pos = generate_benchmark_graph(500,0.1)
# G = tmp_G
InfoNode()
MapEquation()
pos = nx.spring_layout(G)
im = Infomap()

# G = nx.Graph()
# G.add_node(1)
# G.add_node(2)
# G.add_edge(1,2)
for e in G.edges(data='weight', default=1):
    # print(e)
    im.addLink(e[0], e[1], e[2])
im.run()

print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
print("\n#node module")
result = {node.node_id: node.module_id-1 for node in im.tree if node.is_leaf}
infomap_partition = dict(sorted(result.items())) 
# infomap_partition = dict(sorted(result.items())) 

codelength, index_codelength, module_codelength = map_equation(G, infomap_partition)

print("")
print("Result")
print(f"Calculated {codelength} = {index_codelength} + {module_codelength}")
print(f"Correct is {im.codelengths[0]} = {im.index_codelength} + {im.module_codelength}")
print(f"Difference is {im.codelengths[0]-codelength}")
visualize_benchmark_graph(G, pos, infomap_partition)


# %%
# from algorithms.genetic_algorithm import GCM
# G, pos = generate_benchmark_graph(500,0.1)

# genetic_algorithm = GCM(100, 10, fitness_func=community_louvain.modularity)
# gcm_partition_map, gcm_community_map = genetic_algorithm.gcm(G)

# genetic_algorithm = GCM(100, 10, fitness_func=map_equation_wrapper)
# gcm_partition_map_ME, gcm_community_map_ME = genetic_algorithm.gcm(G)

# genetic_algorithm = GCM(100, 10, fitness_func=coverage_wrapper)
# gcm_partition_map_C, gcm_community_map_C = genetic_algorithm.gcm(G)

# genetic_algorithm = GCM(100, 10, fitness_func=performance_wrapper)
# gcm_partition_map_P, gcm_community_map_P = genetic_algorithm.gcm(G)

# %%
# true_partition_map, communities = extract_true_communities(G)

# fig, ax = plt.subplots(3, 2)
# fig.suptitle("NMI values")
# fig.set_size_inches(10, 10)
# ax[0][0].set_title(f"Initial Graph", fontsize=10)
# ax[0][0].set_axis_off()
# ax[0][1].set_title(f"Ground-Truth: {normalized_mutual_information(true_partition_map, true_partition_map)}" , fontsize=10)
# ax[0][1].set_axis_off()
# ax[1][0].set_title(f"Modularity: {normalized_mutual_information(true_partition_map, gcm_partition_map)}" , fontsize=10)
# ax[1][0].set_axis_off()
# ax[1][1].set_title(f"Map-Equation: {normalized_mutual_information(true_partition_map, gcm_partition_map_ME)}" , fontsize=10)
# ax[1][1].set_axis_off()
# ax[2][0].set_title(f"Coverage: {normalized_mutual_information(true_partition_map, gcm_partition_map_C)}" , fontsize=10)
# ax[2][0].set_axis_off()
# ax[2][1].set_title(f"Nothing", fontsize=10)
# ax[2][1].set_axis_off()
# visualize_benchmark_graph(G, pos, None, ax[0][0])
# visualize_benchmark_graph(G, pos, true_partition_map, ax[0][1])
# visualize_benchmark_graph(G, pos, gcm_partition_map, ax[1][0])
# visualize_benchmark_graph(G, pos, gcm_partition_map_ME, ax[1][1])
# visualize_benchmark_graph(G, pos, gcm_partition_map_C, ax[2][0])
# plt.tight_layout()
# plt.show()
# visualize_benchmark_graph(G, pos, gcm_partition_map_ME)

# %%
# print(
#     normalized_mutual_information(true_partition_map, gcm_partition_map),
#     normalized_mutual_information(true_partition_map, gcm_partition_map_ME),
#     normalized_mutual_information(true_partition_map, infomap_partition),
# )


# %%
# %%time
class LouvainCoreAlgorithm:
    
    
    fitness_function = None
    levels = []
    level_fitness = [] 
    null_fitness = []
    level_graphs = []
    # G = None

    def __init__(
        self, 
        max_iter=-1, 
        fitness_function=community_louvain.modularity, 
        verbose=False,
        max_local_movements=sys.maxsize,
        stop_after=-1
        ):
        super().__init__()
        self.fitness_function = fitness_function
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_local_movements = max_local_movements
        # self.run_iteration.__name__ = fitness_function.__name__
        print(f"Inititalizing algorithm with {self.run_iteration.__name__}")
    
    def initialize(self, G):
        initial_partition_map = dict(enumerate(G.nodes()))
        self.levels = []
        self.stats = {
            "local_moving":[]
        }
        self.levels.append(initial_partition_map)
        initial_fitness = self.fitness_function(initial_partition_map, G)
        self.null_fitness.append(initial_fitness)
        self.level_fitness.append(initial_fitness)
        self.level_graphs.append(G)
        self.gain_stats = []
        return G, initial_partition_map


    def run(self, G):
        self.G = G
        _, initial_partition_map = self.initialize(self.G)
        result = self.run_iteration(G, initial_partition_map, self.max_iter)
        if self.verbose:
            print(f"Final results are in! Algorithm found {len(np.unique(list(result.values())))} communities")
        backtracked_partitioning = self.decode_partition_map(len(self.levels)-1)
        return backtracked_partitioning

    def run_iteration(self, G, initial_partition_map, max_iter=-1):
        tmp_partition_map = initial_partition_map.copy()
        tmp_G = G

        while True:
            new_partition_map, final_fitness = self.local_movement(tmp_G, tmp_partition_map)
            if max_iter == 0:
                print(F"STOP: User defined stop after {self.max_iter} iterations")
                break
            if new_partition_map == tmp_partition_map:
                print(f"STOP: Both community_maps are the same -> {new_partition_map == tmp_partition_map}")    
                break

            self.levels.append(new_partition_map)
            self.level_fitness.append(final_fitness)
            self.level_graphs.append(tmp_G)
            print(f"Achieved improvement of {final_fitness - self.level_fitness[-1]} - Starting Next round!")
            new_G, reduced_partition = self.reduce_network(tmp_G, new_partition_map)
            tmp_G = new_G
            tmp_partition_map = reduced_partition
            max_iter -= 1

        return tmp_partition_map

    def local_movement(self, G, partition_map):

        partition_map_copy = partition_map.copy()
        partition_map_result = None
        initial_fitness = self.fitness_function(partition_map_copy, G)
        container_of_n_last_gains = deque(maxlen=10)
        container_of_n_last_gains.append(np.absolute(initial_fitness))
        cnt = 0
        has_improvement = True

        last_improvement = np.absolute(initial_fitness)
        while True and len(G.nodes()) > 1:
            random_order = np.random.permutation(G.nodes()) 
            had_improvement = False
            for node in random_order:
                start = time.time()
                current_communities = np.unique(list(partition_map_copy.values()))
                empty_community = next(iter(set(range(min(current_communities), max(current_communities)+2)) - set(current_communities)))
                candidates = [partition_map_copy[adjacent_node] for adjacent_node in G[node]] + [empty_community]
                gains = [
                    self._compute_fitness(G, partition_map_copy, initial_fitness, node, candidate_community) 
                    for candidate_community 
                    in candidates
                    if partition_map_copy[node] is not candidate_community
                    ] 
                maximum_gain = max(gains, key=operator.itemgetter(1))
                
                end = time.time()
                time_in_sec = end - start
                self.stats["local_moving"].append({"Gain":maximum_gain[1], "Fitness":maximum_gain[2], "ProcessTimeSec": time_in_sec, "Rolling_Avg": None})
                
                
                if maximum_gain[1] > 0:

                    rolling_average = np.mean(container_of_n_last_gains)
                    self.stats["local_moving"][-1]["Rolling_Avg"] = rolling_average

                    had_improvement = True
                    verbose_str2 = f"Increase {initial_fitness:.8f} -> {maximum_gain[2]:.8f} - {time_in_sec:.2f} sec"
                    verbose_str3 = f"Moved node {maximum_gain[3]} to community {maximum_gain[4]}"

                    partition_map_copy = maximum_gain[0]
                    last_improvement = maximum_gain[1]
                    initial_fitness = maximum_gain[2]
                    container_of_n_last_gains.append(maximum_gain[1])
                    cnt=0
                    if self.verbose: print(f"{verbose_str2} | {verbose_str3}")
                cnt += 1
            if had_improvement == False:
                print("No further improvement!")
                break

            if (cnt % 10) == 0:
                print(f"{cnt} node checks without improvement") 
            if cnt > self.max_local_movements:
                print(f"Max iteration reached!")
                break      
        print(f"--- Local movement completed with {initial_fitness} ---\n")      
            

        return partition_map_copy, last_improvement

    def reduce_network(self, G, partition_map):
        print("Start reduction")
        start = time.time()

        communities = np.unique(list(partition_map.values()))
        tmp_G = nx.Graph()
        tmp_G.add_nodes_from(communities)
        edge_accumulator = Counter()
        for node, community in partition_map.items():
            adjacent_partitions = [partition_map[adjacent] for adjacent in G[node]] 
            new_edges = itertools.product([community], adjacent_partitions)
            if self.verbose: print(f"Node {node}: Community {community} connectected to {len(adjacent_partitions)} other communities")
            edge_accumulator.update(new_edges)

        ebunch = [key+({'weight':value},) for key, value in edge_accumulator.most_common()] # TODO
        tmp_G.add_edges_from(ebunch)
        new_partition_map = {node:idx for idx, node in enumerate(tmp_G.nodes())}
        end = time.time()
        print(f"Took {start-end:.2f} seconds to generate the reduced graph")
        return tmp_G, new_partition_map
    
    def decode_partition_map(self, starting_level):
        if starting_level == 0:
            return self._sort_partition_map(self.levels[0])
        if starting_level == 1:
            return self._sort_partition_map(self.levels[1])
        result = self._decode_levels(starting_level-1, self.levels[starting_level])

        return self._sort_partition_map(result)
    
    def _sort_partition_map(self, partition_map):
        return dict(sorted(partition_map.items()))

    def _decode_levels(self, level, subset):
        partitions_from_level = self.levels[level]
        if level == 0:
            result = subset
            return  result
        keys = np.unique(list(subset.keys()))
        result = {}
        for key in keys:
            partition_subset = {node:comm for node, comm in partitions_from_level.items() if comm==key}
            nodes = self._decode_levels(level-1, partition_subset)

            renamed_nodes = {node:key for node, prev_community in nodes.items()}
            result.update(renamed_nodes)
        return result


    def _extract_partition_map(self, communities):
        node_community_participation = {node:idx for idx, community in enumerate(communities) for node in community}
        return OrderedDict(sorted(node_community_participation.items()))

    def _extract_community_map(self, partition):
        v = {}
        for key, value in partition.items():
            v.setdefault(value, []).append(key)
        communities = list(dict(sorted(v.items())).values())
        return communities

    def _compute_fitness(self, G, partition, old_fitness, node, community):
        partition_copy = partition.copy()
        partition_copy[node] = community 
        fitness = self.fitness_function(partition_copy, G)
        gain = fitness - old_fitness
        return partition_copy, gain, fitness, node, community        

G = nx.karate_club_graph()
# G = nx.barbell_graph(5, 3)
# # G = nx.bull_graph()
# # G = nx.generators.erdos_renyi_graph(10, 0.5)
# # G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(4, 20, p_in=0.9, p_out=0.1)
# G, pos = generate_benchmark_graph(250,0.3)
pos = nx.spring_layout(G)

# G, pos = generate_benchmark_graph(250,0.1)
print("Generated network")
louvain_algorithm = LouvainCoreAlgorithm(
    fitness_function=coverage_wrapper, verbose=True)
my_prt = louvain_algorithm.run(G)
true_prt = community_louvain.best_partition(G)
true_partition_map = community_louvain.best_partition(G)

num_plots = len(louvain_algorithm.levels)+1
fig, ax = plt.subplots(num_plots, 1)
fig.set_size_inches(5, 5*num_plots)
for i in range(0,num_plots-1):
    tmp_prt = louvain_algorithm.decode_partition_map(i)
    ax[i].set_title(f"Level {i} -> NMI: {normalized_mutual_information(true_partition_map, tmp_prt)} ", fontsize=10)
    ax[i].set_axis_off()
    visualize_benchmark_graph(G, pos, tmp_prt, ax[i])

ax[-1].set_title(f"True Partition -> NMI: {normalized_mutual_information(true_partition_map, true_prt)}", fontsize=10)
ax[-1].set_axis_off()
visualize_benchmark_graph(G, pos, true_prt, ax[-1])
# %%
# num_plots = len(louvain_algorithm.levels)+1
# fig, ax = plt.subplots(num_plots, 2)
# fig.set_size_inches(10, 15)

# ax[0][0].set_title(f"Graph", fontsize=10)
# ax[0][0].set_axis_off()
# ax[0][1].set_title(f"True Partition -> NMI: {normalized_mutual_information(true_partition_map, true_prt)}", fontsize=10)
# ax[0][1].set_axis_off()
# init_pos = nx.spring_layout(G)
# visualize_benchmark_graph(G, init_pos, None, ax[0][0])
# visualize_benchmark_graph(G, init_pos, true_prt, ax[0][1])
# for cnt in range(1,num_plots-1):
#     i = cnt-1
#     tmp_G = louvain_algorithm.level_graphs[i]
#     tmp_level_prt = louvain_algorithm.levels[i+1]
#     backtracked = louvain_algorithm.decode_partition_map(i)
#     pos = nx.spring_layout(tmp_G)

#     # print(i)
#     # print(tmp_G.nodes())

#     # print(tmp_level_prt)
#     # print(backtracked)
#     # print("HEEEEEEEEEEEEEE")
#     ax[cnt][0].set_title(f"Level {i} -> NMI: {louvain_algorithm.level_fitness[i]} ", fontsize=10)
#     ax[cnt][0].set_axis_off()
#     visualize_benchmark_graph(tmp_G, pos, tmp_level_prt, ax[cnt][0])
#     # print("HAAAAAAAAAAAAAA")
#     ax[cnt][1].set_title(f"Level {i} -> NMI: {normalized_mutual_information(true_partition_map, backtracked)} ", fontsize=10)
#     ax[cnt][1].set_axis_off()
#     visualize_benchmark_graph(G, init_pos, backtracked, ax[cnt][1])

# tmp_G = louvain_algorithm.level_graphs[-1]
# tmp_level_prt = louvain_algorithm.levels[i+1]
# backtracked = louvain_algorithm.decode_partition_map(i)

# ax[-1][0].set_title(f"Level {i} -> NMI: {louvain_algorithm.level_fitness[i]} ", fontsize=10)
# ax[-1][0].set_axis_off()
# visualize_benchmark_graph(tmp_G, pos, tmp_level_prt, ax[-1][0])
# # print("HAAAAAAAAAAAAAA")
# ax[-1][1].set_title(f"Level {i} -> NMI: {normalized_mutual_information(true_partition_map, backtracked)} ", fontsize=10)
# ax[-1][1].set_axis_off()
# visualize_benchmark_graph(G, pos, backtracked, ax[-1][1])

# plt.tight_layout()
# plt.show()

# visualize_benchmark_graph(G, pos, my_prt, ax[-1])
# # %%
# G, pos = generate_benchmark_graph(250,0.1)
# init_pos = nx.spring_layout(G)
# louvain_algorithm = LouvainCoreAlgorithm(fitness_function=coverage_wrapper, verbose=True, max_iter=20)
# my_prt = louvain_algorithm.run(G)
# # visualize_benchmark_graph(G, pos, my_prt)
# #%%
# true_partition_map = community_louvain.best_partition(G)
# num_plots = len(louvain_algorithm.levels)+1
# fig, ax = plt.subplots(num_plots, 1)
# fig.set_size_inches(5, 5*num_plots)
# for i in range(0,num_plots-1):
#     tmp_prt = louvain_algorithm.decode_partition_map(i)
#     ax[i].set_title(f"Level {i} -> NMI: {normalized_mutual_information(true_partition_map, tmp_prt)} ", fontsize=10)
#     ax[i].set_axis_off()
#     visualize_benchmark_graph(G, init_pos, tmp_prt, ax[i])

# ax[-1].set_title(f"True Partition -> NMI: {normalized_mutual_information(true_partition_map, true_prt)}", fontsize=10)
# ax[-1].set_axis_off()
# visualize_benchmark_graph(G, init_pos, true_prt, ax[-1])


# # %%
# louvain_algorithm = LouvainCoreAlgorithm(fitness_function=modularity_wrapper, verbose=True, max_iter=20)
# infomap_algorithm = LouvainCoreAlgorithm(fitness_function=map_equation_wrapper, verbose=True, max_iter=20)
# coverage_algorithm = LouvainCoreAlgorithm(fitness_function=coverage_wrapper, verbose=True, max_iter=20)
# # lblprob_algorithm = post_transform(algorithms.asyn_lpa_communities)

# algorithms_for_experiment = {
#     infomap_algorithm.run:"Map Equation",
#     louvain_algorithm.run: "Louvain",
#     coverage_algorithm.run:"Coverage Maximization",
#     # lblprob_algorithm:"Louvain Algorithm",
#     }
# collected_data = []
# iterations = list(range(0, 1))
# node_sizes = [250]
# mus = np.arange(0.1, 0.6, 0.1)
# configuration_set = itertools.product(*[iterations, algorithms_for_experiment.items(), node_sizes, mus])

# for configuration in configuration_set:
#     collected_data.append(compute_experiment(configuration))

# data = pd.DataFrame(collected_data)
# data
# # %%
# draw_plots(data)

# %%
class HierarchicalLabelPropagation(LouvainCoreAlgorithm):
    def initialize(self, G):
        initial_partition_map = dict(enumerate(G.nodes()))
        self.levels = []
        self.stats = {
            "local_moving":[]
        }
        self.levels.append(initial_partition_map)
        # initial_fitness = self.fitness_function(initial_partition_map, G)
        # self.null_fitness.append(initial_fitness)
        self.level_fitness.append(0)
        self.level_graphs.append(G)
        self.gain_stats = []
        return G, initial_partition_map

    def local_movement(self, G, partition_map):
        # partition_map_copy = partition_map.copy()
        print("")
        print("INPUT")
        print(partition_map)
        num_changes = 0
        partitions = np.unique(list(partition_map.values()))
        num_partitions = len(partitions)
        num_nodes = len(np.unique(list(partition_map.keys())))
        
        A = nx.adjacency_matrix(G).todense()
        
        node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
        id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        id2comm = dict(enumerate(comm2id.keys()))  
        partition_matrix = np.zeros((num_nodes, num_partitions))
        partition_map_copy = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        initial_labels = np.array(list(partition_map.values()))
        # print("Initial nodes")
        # print(G.nodes())
        # while True:
            # random_order = np.random.permutation(G.nodes()) 
            # had_improvement = False
            # for node in random_order:            
        for node, community in np.random.permutation(list(partition_map_copy.items())):
            adj_partitions = [
                    partition_map_copy[node2id[adjacent_node]] 
                    for adjacent_node 
                    in G[id2node[node]]]  

            self_partition = [partition_map_copy[node]]
            chosen = np.random.choice(adj_partitions + self_partition)
            if self.verbose: print(f"Node {node} moved: {community} -> {chosen}")
            partition_map_copy[node] = chosen 

        new_labels = np.array(list(partition_map_copy.values()))
        changes = initial_labels != new_labels
        initial_labels = new_labels
        num_changes = changes.sum()





        resulting_map = {id2node[node]: id2comm[community] for node, community in partition_map_copy.items()}
        # sub_groups = {}
        # for prt in np.unique(list(resulting_map.values())):
        #     sub_prt = {node:community for node, community in resulting_map.items() if community == prt}
        #     sub_G = G.subgraph(list(sub_prt.keys()))
        #     new_division = self.sub_divide(sub_G, sub_prt)
        #     sub_groups.update({node: f"{prt}_{community}" for node, community in new_division.items()})
        # # print(sub_groups)
        # unique_sub_groups = set(sub_groups.values())
        # new_mapping = dict(zip(unique_sub_groups, range(len(unique_sub_groups))))
        # # print(new_mapping)
        # flattenend_map = {node:new_mapping[sub_community] for node, sub_community in sub_groups.items()}
        # resulting_map = flattenend_map
        # print("OUTPUT")
        # print(resulting_map)
        print(f"Number of changes {num_changes}")
        return resulting_map, num_changes

    def sub_divide(self, G, partition_map):
        num_changes = 0
        partition_map = {node: idx for idx, node in enumerate(partition_map)}
        partitions = np.unique(list(partition_map.values()))
        num_partitions = len(partitions)
        num_nodes = len(np.unique(list(partition_map.keys())))
        
        node2id = dict({node: idx for idx, node in enumerate(G.nodes())})
        id2node = dict(enumerate(node2id.keys()))
        comm2id = dict({community: idx for idx, community in enumerate(set(partition_map.values()))})
        id2comm = dict(enumerate(comm2id.keys()))  
        partition_matrix = np.zeros((num_nodes, num_partitions))
        partition_map_copy = {node2id[node]: comm2id[community] for node, community in partition_map.items()}
        initial_labels = np.array(list(partition_map.values()))



        for node, community in np.random.permutation(list(partition_map_copy.items())):
            adj_partitions = [
                    partition_map_copy[node2id[adjacent_node]] 
                    for adjacent_node 
                    in G[id2node[node]]]  

            self_partition = [partition_map_copy[node]]
            chosen = np.random.choice(adj_partitions + self_partition)
            partition_map_copy[node] = chosen 

        new_labels = np.array(list(partition_map_copy.values()))
        changes = initial_labels != new_labels
        initial_labels = new_labels
        num_changes = changes.sum()
        resulting_map = {id2node[node]: id2comm[community] for node, community in partition_map_copy.items()}
        
        return resulting_map
    
    def _randargmax(self, b,**kw):
        """ a random tie-breaking argmax"""
        return np.argmax(np.random.random(b.shape) * (b==b.max(axis=1)), **kw)

G = nx.karate_club_graph()
# G = nx.barbell_graph(5, 3)
# # G = nx.bull_graph()
# # G = nx.generators.erdos_renyi_graph(10, 0.5)
# # G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(4, 50, p_in=0.9, p_out=0.1)
# G, pos = generate_benchmark_graph(250,0.1)
pos = nx.spring_layout(G)

# G, pos = generate_benchmark_graph(250,0.1)
print("Generated network")
init_pos = nx.spring_layout(G)
louvain_algorithm = HierarchicalLabelPropagation(
    fitness_function=None, verbose=False)
my_prt = louvain_algorithm.run(G)
true_prt = community_louvain.best_partition(G)
true_partition_map = community_louvain.best_partition(G)

# %%
num_plots = len(louvain_algorithm.levels)+1
fig, ax = plt.subplots(num_plots, 1)
fig.set_size_inches(5, 5*num_plots)
for i in range(0,num_plots-1):
    tmp_prt = louvain_algorithm.decode_partition_map(i)
    ax[i].set_title(f"Level {i} -> NMI: {normalized_mutual_information(true_partition_map, tmp_prt)} ", fontsize=10)
    ax[i].set_axis_off()
    visualize_benchmark_graph(G, init_pos, tmp_prt, ax[i])

ax[-1].set_title(f"True Partition -> NMI: {normalized_mutual_information(true_partition_map, true_prt)}", fontsize=10)
ax[-1].set_axis_off()
visualize_benchmark_graph(G, init_pos, true_prt, ax[-1])
# %%
# G = nx.karate_club_graph()
G = nx.barbell_graph(5, 3)
# # G = nx.bull_graph()
# # G = nx.generators.erdos_renyi_graph(10, 0.5)
# # G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(4, 50, p_in=0.9, p_out=0.1)
G, pos = generate_benchmark_graph(250,0.7)
pos = nx.spring_layout(G)
import pquality
import pquality.PartitionQuality as pq
true_partition_map = community_louvain.best_partition(G)
communities = extract_community_map(true_partition_map)
scores = pquality.pquality_summary(G, communities)
# scores2 = pquality.pquality_summary(G, extract_community_map({node:0 for node in G.nodes()}))
print(scores['Indexes'])
# print(scores2["Indexes"])
# print(scores['Modularity'])

def flake_odf_wrapper(partition_map, G):
    partitions = np.unique(list(partition_map.values()))
    partition_indegrees = defaultdict(float)
    partition_degrees = defaultdict(float)
    node_indegrees = {}
    node_exdegrees = {}
    node_diff = {}
    for node, community in partition_map.items():
        node_indegrees[node] = sum(True for adj_node in G[node] if partition_map[adj_node] == community)
        node_exdegrees[node] = len(G[node])/2
        node_diff[node] = node_indegrees[node] < node_exdegrees[node]
        partition_indegrees[community] += int(node_diff[node])
        partition_degrees[community] += 1

    fraction = np.array(list(partition_indegrees.values()))/np.array(list(partition_degrees.values()))
    return -fraction.mean()

print(flake_odf_wrapper(true_partition_map, G))
print(flake_odf_wrapper(dict(enumerate(G)), G))
print(flake_odf_wrapper({node:0 for node in G.nodes()}, G))
# print(flake_odf_wrapper({node:0 for node in G.nodes()}, G))
# tmp_prt = flake_odf_wrapper.copy().setdefault(0, 1)



# %%
print("Generated network")
init_pos = nx.spring_layout(G)
louvain_algorithm = LouvainCoreAlgorithm(
    fitness_function=flake_odf_wrapper, verbose=True)
my_prt = louvain_algorithm.run(G)
true_prt = community_louvain.best_partition(G)
true_partition_map = community_louvain.best_partition(G)


#%%
num_plots = len(louvain_algorithm.levels)+1
fig, ax = plt.subplots(num_plots, 1)
fig.set_size_inches(5, 5*num_plots)
for i in range(0,num_plots-1):
    tmp_prt = louvain_algorithm.decode_partition_map(i)
    ax[i].set_title(f"Level {i} -> NMI: {normalized_mutual_information(true_partition_map, tmp_prt)} ", fontsize=10)
    ax[i].set_axis_off()
    visualize_benchmark_graph(G, init_pos, tmp_prt, ax[i])

ax[-1].set_title(f"True Partition -> NMI: {normalized_mutual_information(true_partition_map, true_prt)}", fontsize=10)
ax[-1].set_axis_off()
visualize_benchmark_graph(G, init_pos, true_prt, ax[-1])

# %%
# %%
# G = nx.karate_club_graph()
# G = nx.barbell_graph(5, 2)
# # G = nx.bull_graph()
# # G = nx.generators.erdos_renyi_graph(10, 0.5)
# # G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(4, 50, p_in=0.9, p_out=0.3)
G, pos = generate_benchmark_graph(250,0.2)
pos = nx.spring_layout(G)
import pquality
import pquality.PartitionQuality as pq
true_partition_map = community_louvain.best_partition(G)
lpa_prt = extract_partition_map(algorithms.asyn_lpa_communities(G))
communities = extract_community_map(true_partition_map)
scores = pquality.pquality_summary(G, communities)
scores2 = pquality.pquality_summary(G, extract_community_map(lpa_prt))
print(scores['Indexes'])
print(scores2["Indexes"])
print(scores['Modularity'])
print(scores2['Modularity'])

def conductance_wrapper(partition_map, G):
    partitions = np.unique(list(partition_map.values()))
    partition_edges = defaultdict(float)
    partition_nodes = defaultdict(float)
    node_edges = {}
    for node, community in partition_map.items():
        node_edges[node] = sum(True for adj_node in G[node] if partition_map[adj_node] != community)
        partition_nodes[community] += int(node_edges[node])
        partition_edges[community] += int(len(G[node])) - int(node_edges[node])

    cs = np.array(list(partition_nodes.values()))
    ms = np.array(list(partition_edges.values()))

    fraction = cs/((ms)+cs)

    return -fraction.mean()

print(conductance_wrapper(true_partition_map, G))
print(conductance_wrapper(lpa_prt, G))
visualize_benchmark_graph(G, pos, lpa_prt)
# print(conductance_wrapper(dict(enumerate(G)), G))
# conductance_wrapper({node:0 for node in G.nodes()}, G)

# %%
