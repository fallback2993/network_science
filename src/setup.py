#%%
import sys
!conda install --yes --prefix {sys.prefix} matplotlib pandas scikit-learn scipy networkx
# !conda install --yes --prefix {sys.prefix} -c conda-forge/label/cf202003 infomap
!{sys.executable} -m pip install python-louvain multiprocess wurlitzer
!{sys.executable} -m pip install -e vendor/py
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
from collections import OrderedDict, Counter
import pandas as pd
import multiprocess as mp
import matplotlib.cm as cm
import community as community_louvain
import scipy

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
    node_community_participation = {node:idx for idx, community in enumerate(communities) for node in community}
    return OrderedDict(sorted(node_community_participation.items()))

def extract_community_map(partition):
    v = {}
    for key, value in partition.items():
        v.setdefault(value, set()).add(key)
    communities = dict(sorted(v.items())).values()
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
node_sizes = [250, 500]
mus = np.arange(0.1, 1.1-0.5, 0.1)
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

    fig.delaxes(*axes[num_groups:])
    fig.set_tight_layout(True)
    return plt.show()

draw_plots(data)
# %%
def modularity_wrapper(G, partition):
    return community_louvain.modularity(partition, G)

modularity_wrapper(G, lpa_partition_map)
# %%
def map_equation_wrapper(G, partition):
    # adjacency_matrix = nx.adjacency_matrix(G)
    # left_eigs = scipy.linalg.eig(adjacency_matrix.A, left=True)
    # print(adjacency_matrix.A)
    # print(left_eigs[0])
    numerical_stabilizer = np.finfo(float).eps
    transition_matrix = nx.algorithms.google_matrix(G)
    initial_starting_point = np.ones((transition_matrix.shape[0],1))/transition_matrix.shape[0]
    A = transition_matrix.T
    eigen_vector = initial_starting_point
    for k in range(1000):
        normalizer = np.linalg.norm(A * eigen_vector)
        eigen_vector = A * eigen_vector / normalizer
        # print(f"{k}. iteration")
        # print(b_k)
        # print(normalizer)
    stationary_node_distribution = np.array(eigen_vector/sum(eigen_vector)).squeeze()
    # print(stationary_node_distribution)
    eigen_value = normalizer + numerical_stabilizer
    community_map = extract_community_map(partition)
    # print(community_map)

    adjacent_nodes_per_node = {
        node : (partition[node],[partition[adjacent_node] for adjacent_node in G[node]]) 
        for idx, module in enumerate(community_map) 
        for node in module
    }
    num_partitions = max(set(partition.values()))+1
    partition_exit_links = np.zeros(num_partitions)
    partition_links_sums = np.zeros(num_partitions)
    partition_probabilities = np.zeros(num_partitions)
    for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items():
        partition_exit_links[node_partition] += sum(node_partition != np.array(neighbor_partitions)) 
        partition_links_sums[node_partition] += len(neighbor_partitions)
        partition_probabilities[node_partition] += stationary_node_distribution[node]
    # tmp = {node:(sum([partition[node] != partition[neighbor] for neighbor in adj]))  for node, adj in adjacent_nodes_per_node.items()}
    # tmp2 = {len(adj) for node, adj in adjacent_nodes_per_node.items()}
    # partition_exit_prob = {}
    # for node, prob in tmp.items():
    #     partition_exit_prob[partition[node]] = partition_exit_prob.setdefault(partition[node], 0) + prob
    partition_exit_prob = np.nan_to_num(partition_exit_links/partition_links_sums)
    node_exit_prob = {node: sum(node_partition != np.array(neighbor_partitions)) for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items()}
    # print("CM")
    # print(community_map)
    # print("")
    # print("APN")
    # print(adjacent_nodes_per_node)
    # print("")
    # print("partition_exit_links")
    # print(partition_exit_links)
    # print("")
    # print("partition_links_sums")
    # print(partition_links_sums)
    # print("")
    # print("partition exit probs")
    # print(partition_exit_prob)
    # print("")
    # print("partition probs")
    # print(partition_probabilities)

    # print(sum(stationary_node_distribution))

    w_out = partition_exit_prob.sum()
    w_i_out = partition_exit_prob 
    w_a = stationary_node_distribution
    w_i = partition_probabilities
    w_i_out_add_w_i = w_i_out + w_i 

    term1 = w_out * np.nan_to_num(np.log2(w_out))
    term2 = 2 * w_i_out.dot(np.nan_to_num(np.log2(w_i_out)))
    term3 = w_a.dot(np.nan_to_num(np.log2(w_a)))
    term4 = w_i_out_add_w_i.dot(np.nan_to_num(np.log2(w_i_out_add_w_i)))

    print(term1, term2, term3, term4)
    L = term1 - term2 - term3 + term4
    L = np.asarray(L).flatten()[0]
    return L


# # %%
# G = nx.karate_club_graph()
# true_partition_map = community_louvain.best_partition(G)
# communities = algorithms.greedy_modularity_communities(G)
# cnm_partition_map = extract_partition_map(communities)
# map_equation_wrapper(G, cnm_partition_map)
# map_equation_wrapper(G, true_partition_map)
#%%
def map_equation_wrapper(G, partition):
    # adjacency_matrix = nx.adjacency_matrix(G)
    # left_eigs = scipy.linalg.eig(adjacency_matrix.A, left=True)
    # print(adjacency_matrix.A)
    # print(left_eigs[0])
    numerical_stabilizer = np.finfo(float).eps
    transition_matrix = nx.algorithms.google_matrix(G)
    initial_starting_point = np.ones((transition_matrix.shape[0],1))/transition_matrix.shape[0]
    A = transition_matrix.T
    eigen_vector = initial_starting_point
    for k in range(1000):
        normalizer = np.linalg.norm(A * eigen_vector)
        eigen_vector = A * eigen_vector / normalizer
        # print(f"{k}. iteration")
        # print(b_k)
        # print(normalizer)
    stationary_node_distribution = np.array(eigen_vector/sum(eigen_vector)).squeeze()
    # print(stationary_node_distribution)
    eigen_value = normalizer + numerical_stabilizer
    community_map = extract_community_map(partition)
    # print(community_map)

    adjacent_nodes_per_node = {
        node : (partition[node],[partition[adjacent_node] for adjacent_node in G[node]]) 
        for idx, module in enumerate(community_map) 
        for node in module
    }
    num_partitions = max(set(partition.values()))+1
    partition_exit_links = np.zeros(num_partitions)
    partition_links_sums = np.zeros(num_partitions)
    partition_probabilities = np.zeros(num_partitions)
    for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items():
        partition_exit_links[node_partition] += sum(node_partition != np.array(neighbor_partitions)) 
        partition_links_sums[node_partition] += len(neighbor_partitions)
        partition_probabilities[node_partition] += stationary_node_distribution[node]
    # tmp = {node:(sum([partition[node] != partition[neighbor] for neighbor in adj]))  for node, adj in adjacent_nodes_per_node.items()}
    # tmp2 = {len(adj) for node, adj in adjacent_nodes_per_node.items()}
    # partition_exit_prob = {}
    # for node, prob in tmp.items():
    #     partition_exit_prob[partition[node]] = partition_exit_prob.setdefault(partition[node], 0) + prob
    partition_exit_prob = np.nan_to_num(partition_exit_links/partition_links_sums)
    node_exit_prob = {node: sum(node_partition != np.array(neighbor_partitions)) for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items()}

    q_out = partition_exit_prob.sum()
    q_i_out = partition_exit_prob 
    p_i_in = partition_probabilities
    denominator = p_i_in + q_i_out
    p_i_out_sum_p_i = stationary_node_distribution

    H_Q = - (q_i_out/q_out).dot(np.nan_to_num(np.log2(q_i_out/q_out)))
    H_P_i = - np.nan_to_num(q_i_out/denominator) * np.nan_to_num(np.log2(q_i_out/denominator)) - np.nan_to_num(p_i_in/denominator) * np.nan_to_num(np.log2(p_i_in/denominator))

    print(q_out, H_Q)
    print(denominator)
    print(H_P_i)
    # print(q_out * H_Q)
    L = q_out * H_Q + denominator.dot(H_P_i)
    L = np.asarray(L).flatten()[0]
    return L
# %%
from infomap import Infomap
from collections import defaultdict 
# g = convert_graph_formats(g_original, nx.Graph)

g1 = nx.karate_club_graph()
name_map = nx.get_node_attributes(g1, 'club')
coms_to_node = defaultdict(list)

im = Infomap()
for e in g1.edges():
    im.addLink(e[0], e[1])
im.run()

for node in im.iterTree():
    if node.isLeaf():
        nid = node.physicalId
        module = node.moduleIndex()
        nm = name_map[nid]
        coms_to_node[module].append(nm)

coms_infomap = [list(c) for c in coms_to_node.values()]
im.codelengths
print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

print("Result")
print("\n#node module")
result = {node.node_id: node.module_id for node in im.tree if node.is_leaf}
infomap_partition = dict(sorted(result.items())) 
my_codelength = map_equation_wrapper(g1, infomap_partition)
print(my_codelength, im.codelengths[0])
# %%

# %%
