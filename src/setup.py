#%%
import sys
!conda install --yes --prefix {sys.prefix} matplotlib pandas scikit-learn scipy networkx
# !conda install --yes --prefix {sys.prefix} -c conda-forge/label/cf202003 infomap
!{sys.executable} -m pip install python-louvain multiprocess wurlitzer
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

def map_equation(G, partition):
    def compute_weighted_entropy(probs, normalizer):
        return np.nansum((probs/normalizer) * np.log2(probs/normalizer))

    numerical_stabilizer = np.finfo(float).eps
    transition_matrix = nx.algorithms.google_matrix(G)
    initial_starting_point = np.ones((transition_matrix.shape[0],1))/transition_matrix.shape[0]
    A = transition_matrix.T  + numerical_stabilizer
    eigen_vector = initial_starting_point
    normalizer = 0
    for _ in range(1000):
        # print(_)
        normalizer = np.linalg.norm(A * eigen_vector)
        eigen_vector = A * eigen_vector / normalizer
        # print(np.array(eigen_vector/sum(eigen_vector)).sum())
        if math.isclose(normalizer, 1):
            break
    stationary_node_distribution = np.array(eigen_vector/sum(eigen_vector)).squeeze()
    eigen_value = normalizer  # numerical_stabilizer
    
    community_map = dict(enumerate(extract_community_map(partition))) # For some reason partition zero misses
    num_links = len(G.edges())
    num_nodes = len(G.nodes())

    adjacent_nodes_per_node = {
        node : (partition[node],[partition[adjacent_node] for adjacent_node in G[node]]) 
        for name, community in community_map.items() 
        for node in community
    }

    adjacent_partition_matrix = np.full([num_nodes, num_nodes], np.nan)
    for name, community in community_map.items(): 
        for node in community:
            adjacent_partition_matrix[node][node] = partition[node]
            for adjacent_node in G[node]:
                adjacent_partition_matrix[node][adjacent_node] = partition[adjacent_node]
    diagonal = np.diagonal(adjacent_partition_matrix)

    zm = np.ma.masked_where(np.isnan(adjacent_partition_matrix), adjacent_partition_matrix)
    apm_linkage = zm == diagonal[:, None]

    num_partitions = np.max(list(partition.values())) + 1
    partition_ex_links = np.zeros(num_partitions)
    partition_in_links = np.zeros(num_partitions)
    
    node_partition_in_links = np.sum(apm_linkage==True, axis=1)-1
    node_partition_ex_links = np.sum(apm_linkage==False, axis=1)
    
    for partition in np.unique(diagonal):
        partition = int(partition)
        indices_to_check = list(np.where(diagonal == partition)[0])
        partition_ex_links[partition] = sum(node_partition_ex_links[indices_to_check])
        partition_in_links[partition] = sum(node_partition_in_links[indices_to_check])

    adjacent_nodes_per_node = dict(sorted(adjacent_nodes_per_node.items()))

    partition_probabilities = np.zeros(num_partitions)
    partition_links = (partition_in_links+partition_ex_links)
    partition_exit_prob = np.nan_to_num(partition_ex_links/partition_links)/num_partitions

    node_relative_weight = stationary_node_distribution

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


G = generator.planted_partition_graph(5,50, p_in=0.3, p_out=0.01)
map_equation_wrapper(dict(community_louvain.best_partition(G)), G)

# %%

# %%
from infomap import Infomap
from collections import defaultdict 
# g = convert_graph_formats(g_original, nx.Graph)

G = nx.karate_club_graph()
G = nx.barbell_graph(5, 1)
G = nx.bull_graph()
G = nx.generators.erdos_renyi_graph(10, 0.5)
G = nx.generators.cubical_graph()
G = generator.planted_partition_graph(5,50, p_in=0.3, p_out=0.01)
G, pos = generate_benchmark_graph(500,0.1)


pos = nx.spring_layout(G)
im = Infomap()
for e in G.edges():
    im.addLink(e[0], e[1])
im.run()

print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
print("\n#node module")
result = {node.node_id: node.module_id-1 for node in im.tree if node.is_leaf}
infomap_partition = dict(sorted(result.items())) 

codelength, index_codelength, module_codelength = map_equation(G, infomap_partition)

print("")
print("Result")
print(f"Calculated {codelength} = {im.index_codelength} + {im.module_codelength}")
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
class LouvainAlgorithm:
    
    
    fitness_function = None
    levels = []

    def __init__(self, resolution=0.01 ,max_iter=10, fitness_function=community_louvain.modularity, verbose=False):
        self.fitness_function = fitness_function
        self.verbose = verbose
        self.max_iter = max_iter
        self.resolution = resolution
        super().__init__()
    
    def run_louvain(self, G):
        self.G = G
        A = nx.adjacency_matrix(self.G)
        initial_partition_map = dict(enumerate(self.G.nodes()))
        self.levels.append(initial_partition_map)
        result = self.run_iteration(G, initial_partition_map)
        if self.verbose:
            print(f"Final results are in! Algorithm found {len(np.unique(list(result.values())))} communities")
        backtracked_partitioning = self.decode_partition_map(len(self.levels)-1)
        return backtracked_partitioning

    def run_iteration(self, G, initial_partition_map):
        
        new_partition_map = self.local_movement(G, initial_partition_map)
        if self.verbose:
            print(f"Both community_maps are the same -> {new_partition_map == initial_partition_map}")    
        if new_partition_map == initial_partition_map:
            return new_partition_map
        self.levels.append(new_partition_map)
        # print(new_partition_map)
        # return new_partition_map
        final_partition = {}
        community_map = self._extract_community_map(new_partition_map)
        num_communities = len(community_map)
        num_nodes = len(G.nodes())
        # if num_communities < num_nodes:
        reduced_adjacency, reduced_partitions = self.reduce_network(G, new_partition_map)
        
        # return new_partition_map, reduced_adjacency, reduced_partitions
        reduced_community_map = self._extract_community_map(new_partition_map)
        final_partition = self.run_iteration(reduced_adjacency, reduced_partitions)
            # partition_map_old = new_partition_map
            # for i in range(len(num_communities)):
            #     community_map[i] = reduced_community_map[i]
        return final_partition

    def local_movement(self, G, partition_map):
        partition_map_copy = partition_map.copy()
        partition_map_result = None
        initial_fitness = self.fitness_function(partition_map_copy, G)
        fitness = -100
        cnt = 0
        has_improvement = True
        # while has_improvement:
        random_order = np.random.permutation(G.nodes())
        has_improvement = False
        for node in random_order:
            has_improvement = False
            empty_community = np.max(random_order)
            candidates = [partition_map_copy[adjacent_node] for adjacent_node in G[node]] + [empty_community]
            for candidate_community in candidates: 
                curr_community = partition_map_copy[node]
                partition_map_copy[node] = candidate_community 
                fitness = self.fitness_function(partition_map_copy, G)
                if fitness > initial_fitness:
                    if self.verbose: print(f"Increase {initial_fitness:.8f} -> {fitness:.8f}")
                    initial_fitness = fitness
                    has_improvement = True
                    cnt = 0
                    continue
                partition_map_copy[node] = curr_community   
            
            # print(f"New BIGLI {has_improvement} with {fitness} : {initial_fitness}")
            cnt+=1
            if cnt > self.max_iter:
                break

                    # break
            # if cnt > self.max_iter: break         
        print(f"Local movement completed with {initial_fitness}")      
            

        return partition_map_copy

    def reduce_network(self, G, partition_map):
        communities = np.unique(list(partition_map.values()))
        # num_communities = len(communities)
        tmp_G = nx.Graph()
        # print(communities)
        tmp_G.add_nodes_from(communities)
        edge_accumulator = []
        for node, community in partition_map.items():
            adjacent_partitions = [partition_map[adjacent] for adjacent in G[node]] 
            new_edges = list(itertools.product([community], adjacent_partitions))
            edge_accumulator.extend(new_edges)
        # for edge in edge_accumulator:
        counter = Counter(edge_accumulator)
        ebunch = [key+({'weight':value},) for key, value in counter.most_common()]
        # print(ebunch)
        # print(tmp_G.edges())
        tmp_G.add_edges_from(ebunch)
        # print(edge_accumulator)
        new_partition_map = {node:idx for idx, node in enumerate(tmp_G.nodes())}
        return tmp_G, new_partition_map
    
    def decode_partition_map(self, starting_level):
        if starting_level <= 1:
            return self._sort_partition_map(self.levels[0])
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
        # print(result)
        return result


    def _extract_partition_map(self, communities):
        # print(communities)
        node_community_participation = {node:idx for idx, community in enumerate(communities) for node in community}
        return OrderedDict(sorted(node_community_participation.items()))

    def _extract_community_map(self, partition):
        v = {}
        for key, value in partition.items():
            v.setdefault(value, []).append(key)
        communities = list(dict(sorted(v.items())).values())
        return communities

# G = nx.karate_club_graph()
# G = nx.barbell_graph(5, 3)
# G = nx.bull_graph()
# G = nx.generators.erdos_renyi_graph(10, 0.5)
# G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(4,10, p_in=0.9, p_out=0.1)
# G, pos = generate_benchmark_graph(250,0.1)
pos = nx.spring_layout(G)

G, pos = generate_benchmark_graph(250,0.1)
print("Generated network")

louvain_algorithm = LouvainAlgorithm(verbose=True, max_iter=20)
my_prt = louvain_algorithm.run_louvain(G)
true_prt = community_louvain.best_partition(G)

# %%
true_partition_map, communities = extract_true_communities(G)

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
# visualize_benchmark_graph(G, pos, my_prt, ax[-1])

# %%
# print(f"{normalized_mutual_information(true_partition_map, my_prt)} vs. {normalized_mutual_information(true_partition_map, true_prt)}")


# %%
louvain_algorithm.decode_partition_map(1)