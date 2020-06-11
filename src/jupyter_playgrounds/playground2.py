
# %%
import sys
!conda install --yes --prefix {sys.prefix} matplotlib pandas scikit-learn scipy networkx jupyter
# !conda install --yes --prefix {sys.prefix} -c conda-forge/label/cf202003 infomap
!{sys.executable} -m pip install python-louvain multiprocess 
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
from utils import extract_partition_map, extract_community_map, generate_benchmark_graph, visualize_benchmark_graph

import pickle
import torch
from torch import FloatTensor, LongTensor
from typing import Dict, Callable, List
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# https://medium.com/@sddkal/random-walks-on-adjacency-matrices-a127446a6777#:~:text=A%20random%20walk%20is%20a,can%20perform%20a%20random%20walk.
# %%
# G = nx.karate_club_graph()
# G = nx.barbell_graph(5, 2)
# # G = nx.bull_graph()
# # G = nx.generators.erdos_renyi_graph(10, 0.5)
# # G = nx.generators.cubical_graph()
# G = generator.planted_partition_graph(4, 50, p_in=0.9, p_out=0.05)
G, pos = generate_benchmark_graph(250,0.1)
pos = nx.spring_layout(G)
true_partition_map = community_louvain.best_partition(G)
lpa_prt = extract_partition_map(algorithms.asyn_lpa_communities(G))
communities = extract_community_map(true_partition_map)

nx.draw_networkx(G, pos, edgecolors="black", node_size=600, cmap=plt.cm.RdYlBu, node_color=list(true_partition_map.values()))

# %%
def random_walk(a, i, iters):
    # a -> adj
    # i -> starting row
    walk = np.zeros(iters+1) # holds transitions
    walk[0] = i
    elements = np.arange(a.shape[0]) # for our graph [0,1,2,3]
    c_index = i # current index for this iteration
    for k in range(iters):
        count = 0 # count of transitions
        probs = a[c_index].reshape((-1,))  # probability of transitions
        # sample from probs
        sample = np.random.choice(elements,p=probs) # sample a target using probs
        index = sample # go to target
        walk[k+1] = index
        c_index = index
    return walk

# print(pd.DataFrame(nx.adj_matrix(G).todense()))
walk_length = 1000000
markov_matrix = np.array(nx.google_matrix(G, alpha=1))
nodes = G.nodes()
vocab = {f"node_{node}":node for node in nodes}
n2voc = {node:name for name, node in vocab.items()}
starting_point = np.random.choice(nodes)
walk = random_walk(markov_matrix, starting_point, walk_length)
walk


# %%
sliding_windows = np.vstack((walk,np.roll(walk, -1), np.roll(walk, -2), np.roll(walk, -3), np.roll(walk, -4))).astype(np.int)
sliding_windows
# %%
cooccurence_matrix = np.zeros_like(markov_matrix)
center_node_pos = int(sliding_windows.shape[0]/2)
for position in range(walk_length):
    left_left_word = sliding_windows[0, position]
    left_word = sliding_windows[1, position]
    center_word = sliding_windows[2, position]
    right_word = sliding_windows[3, position]
    right_right_word = sliding_windows[4, position]
    cooccurence_matrix[center_word,left_word] += 1
    cooccurence_matrix[center_word,right_word] += 1
    cooccurence_matrix[center_word,left_left_word] += 1
    cooccurence_matrix[center_word,right_right_word] += 1

pd.DataFrame(cooccurence_matrix)

# %%
X = torch.tensor(cooccurence_matrix)
X = X.to(device) # If GPU enabled
X = X.to(torch.float64) + 0.1

def to_probabilities(count_matrix: FloatTensor) -> FloatTensor:
    return count_matrix / torch.sum(count_matrix, dim=1)

P = to_probabilities(X)
P
 # %%
def weight_fn(X: FloatTensor, x_max: int, a: float) -> FloatTensor:
    all_ones = torch.ones_like(X, requires_grad = False)
    temp = torch.pow(X/x_max, 0.75)
    result = torch.where(X<x_max, torch.pow(X/x_max, 0.75), torch.ones_like(X))
    return result

X_weighted = weight_fn(X, 100, 0.75)
X_weighted = X_weighted.to(device)
X_weighted

# %%
def loss_fn(X_weighted: FloatTensor, W: FloatTensor, W_context: FloatTensor, 
            B: FloatTensor, B_context: FloatTensor, 
            X: FloatTensor, device:str = "cpu") -> FloatTensor:
    hypothesis = (torch.mm(W, W_context.transpose(0,1))  + B + B_context).type(torch.DoubleTensor).to(device)
    target = torch.log(X)
    squared_loss = ((hypothesis - target)**2)
    temp = torch.mul(X_weighted, squared_loss)
    result = torch.sum(temp)
    return result

class GloVe(torch.nn.Module):
    def __init__(self, vocab: Dict[str, int], vector_dimensionality: int=30, device: str='cpu') -> None:
        super(GloVe, self).__init__()
        self.device = device
        self.vocab_len = len(vocab)
        self.w = torch.nn.Embedding(num_embeddings = self.vocab_len, embedding_dim=vector_dimensionality).to(self.device)
        self.wc = torch.nn.Embedding(num_embeddings = self.vocab_len, embedding_dim=vector_dimensionality).to(self.device)
        self.b = torch.nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=1).to(self.device)
        self.bc = torch.nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=1).to(self.device)
        
    def forward(self, X_weighted: FloatTensor, X: FloatTensor) -> FloatTensor:
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        W = self.w(embedding_input)
        W_context = self.wc(embedding_input)
        B = self.b(embedding_input)
        B_context = self.bc(embedding_input)
        return loss_fn(X_weighted, W, W_context,B, B_context, X, self.device)
    
    def get_vectors(self) -> FloatTensor:
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        return self.w(embedding_input) + self.wc(embedding_input)

num_epochs = 300
all_losses = []
network = GloVe(vocab=vocab,vector_dimensionality=30, device=device)
opt = torch.optim.Adam(network.parameters(), lr=0.05) 

for i in range(num_epochs):
    loss = network.forward(X_weighted, X) # backward
    value = loss.data.cpu().numpy()
    if i % 20 == 0:
        print(f"Epoch: {i} - Loss is currently at: {loss}")
    all_losses.append(value)
    loss.backward()
    opt.step()
    opt.zero_grad()   

plt.plot(np.array(all_losses))


# %%
def similarity(word_i: str, word_j: str, vocab: Dict[str, int], vectors: FloatTensor) -> float:
    i = vocab[word_i]
    j = vocab[word_j] 
    v_i = vectors[i] / torch.norm(vectors[i], p=2)  # a/|a|
    v_j = vectors[j] / torch.norm(vectors[j], p=2)  # b/|b|
    sim = torch.mm(v_i.view(1, -1), v_j.view(-1, 1)).item()
    return sim


def similarities(word_i: str, vocab: Dict[str, int], vectors: FloatTensor) -> FloatTensor:
    i = vocab[word_i]
    comparison_vector = vectors[i]
    inner_products = torch.mm(comparison_vector.view(1, -1), vectors.transpose(0, 1))
    matrix_norms = torch.norm(vectors, p=2, dim=1)
    comparison_norm = torch.norm(comparison_vector, p=2)
    return inner_products / (comparison_norm * matrix_norms)


def most_similar(word_i: str, vocab: Dict[str, int], vectors: FloatTensor, k: int) -> List[str]:
    sims = similarities(word_i, vocab, vectors)
    _, topi = sims.topk(dim=-1, k=k)
    topi = topi.view(-1).cpu().numpy().tolist()
    inv = {v: i for i, v in vocab.items()}
    return [inv[i] for i in topi if inv[i] != word_i]

word_vectors = network.get_vectors().detach()
for word in ['node_0', 'node_6' ,'node_9', 'node_7']:
    print('Most similar words to {}: {}'.format(word, most_similar(word, vocab, word_vectors, 6)))

# nx.draw_networkx(G, pos, edgecolors="black", node_size=600, cmap=plt.cm.RdYlBu, node_color=list(true_partition_map.values()))

# %%
# for prt in np.unique(list(true_partition_map.values())):
all_pairs = list(itertools.permutations(vocab.keys(),2))
all_pair_scores = list(zip(all_pairs, (similarity(n1, n2, vocab, word_vectors) for n1, n2 in all_pairs)))
# all_pair_scores

# %%
edge_proximities = {}
for n1, n2 in G.edges():
    # print(n1, n2)
    prt = "_".join(set((str(true_partition_map[n1]), str(true_partition_map[n2]))))
    edge = (n1, n2)
    edge_proximities[(n1, n2, prt)] = similarity(n2voc[n1], n2voc[n2], vocab, word_vectors)

# edge_proximities
# %%
partition_aggregation = defaultdict(float)
partition_counts = defaultdict(float)
for (n1, n2, prt), sim in edge_proximities.items():
    partition_aggregation[prt] += sim
    partition_counts[prt] += 1

a_partition_aggregation = np.array(list(partition_aggregation.values()))
a_partition_counts = np.array(list(partition_counts.values()))
print(partition_aggregation.keys())
print(a_partition_aggregation)
print(a_partition_counts)
print(a_partition_aggregation/a_partition_counts)
print(sum(a_partition_aggregation/a_partition_counts))
print(a_partition_aggregation.sum())
print(a_partition_aggregation.sum()/len(G.nodes()))

# %%
random_prt = dict(enumerate(G.nodes()))
edge_proximities = {}
for n1, n2 in G.edges():
    # print(n1, n2)
    prt = "_".join(set((str(random_prt[n1]), str(random_prt[n2]))))
    edge = (n1, n2)
    edge_proximities[(n1, n2, prt)] = similarity(n2voc[n1], n2voc[n2], vocab, word_vectors)

partition_aggregation = defaultdict(float)
partition_counts = defaultdict(float)
for (n1, n2, prt), sim in edge_proximities.items():
    partition_aggregation[prt] += sim
    partition_counts[prt] += 1

a_partition_aggregation = np.array(list(partition_aggregation.values()))
a_partition_counts = np.array(list(partition_counts.values()))
print(partition_aggregation.keys())
print(a_partition_aggregation)
print(a_partition_counts)
print(a_partition_aggregation/a_partition_counts)
print(sum(a_partition_aggregation/a_partition_counts))
print(a_partition_aggregation.sum())
print(a_partition_aggregation.sum()/len(G.nodes()))

# %%
random_prt = dict(itertools.product(G.nodes(), [0]))
edge_proximities = {}
for n1, n2 in G.edges():
    # print(n1, n2)
    prt = "_".join(set((str(random_prt[n1]), str(random_prt[n2]))))
    edge = (n1, n2)
    edge_proximities[(n1, n2, prt)] = similarity(n2voc[n1], n2voc[n2], vocab, word_vectors)

partition_aggregation = defaultdict(float)
partition_counts = defaultdict(float)
for (n1, n2, prt), sim in edge_proximities.items():
    partition_aggregation[prt] += sim
    partition_counts[prt] += 1

a_partition_aggregation = np.array(list(partition_aggregation.values()))
a_partition_counts = np.array(list(partition_counts.values()))
print(partition_aggregation.keys())
print(a_partition_aggregation)
print(a_partition_counts)
print(a_partition_aggregation/a_partition_counts)
print(sum(a_partition_aggregation/a_partition_counts))
print(a_partition_aggregation.sum())
print(a_partition_aggregation.sum()/len(G.nodes()))

# %%

def similarities_to_others(c_vec:FloatTensor, vectors: FloatTensor) -> FloatTensor:
    try:
        comparison_vector = c_vec
        inner_products = torch.mm(comparison_vector.view(1,-1), vectors.transpose(0,1))
        matrix_norms = torch.norm(vectors, p=2, dim=1)
        comparison_norm = torch.norm(comparison_vector, p=2)
        return inner_products/(comparison_norm*matrix_norms)
    except RuntimeError as identifier:
        pass

def most_similar_vector(c_vec:FloatTensor, vocab: Dict[str, int], vectors: FloatTensor, skip:List[int]) -> List[str]:
    sims = similarities_to_others(c_vec, vectors)
    values, indices = sims.sort(descending=True)
    topi = indices.view(-1).cpu().numpy().tolist()
    topv = values.view(-1).cpu().numpy().tolist()
    sorted_mapping = [(val, idx) for val, idx in zip(topv, topi) if idx not in skip]
    inv = {v: i for i, v in vocab.items()}
    most_similar = sorted_mapping[0]
    return inv[most_similar[1]], most_similar[0], most_similar[1]

def update_centroid(centroid, centroid_cnt, added_vector):
    return ((centroid * centroid_cnt) + added_vector)/(centroid_cnt + 1)

# def compute_avg_diff_to_centroid(nodes, word_vectors):
#     if len(nodes) == 0:
#         return torch.zeros_like(word_vectors[[0]]).sum()
#     prt_candidate_vectors = word_vectors[nodes]
#     prt_candidate_centroid = prt_candidate_vectors.sum(axis=0)/prt_candidate_vectors.shape[0]
#     candidate_sum_of_centroid_diff = similarities_to_others(prt_candidate_centroid, prt_candidate_vectors)
#     candidate_avg_centroid_differences = candidate_sum_of_centroid_diff.sum()/candidate_sum_of_centroid_diff.shape[1]
#     return candidate_avg_centroid_differences

def compute_avg_diff_to_centroid(nodes, word_vectors):
    if len(nodes) <= 1:
        return torch.zeros_like(word_vectors[[0]]).sum()
    prt_candidate_vectors = word_vectors[nodes]
    prt_candidate_centroid = prt_candidate_vectors.sum(axis=0)/prt_candidate_vectors.shape[0]
    candidate_sum_of_centroid_diff = torch.dist(prt_candidate_centroid, prt_candidate_vectors)
    candidate_avg_centroid_differences = candidate_sum_of_centroid_diff.sum()/len(nodes)
    return candidate_avg_centroid_differences


random_prt = dict(enumerate(G.nodes()))
unique_partitions = np.unique(list(random_prt.values()))
num_partitions = len(unique_partitions)
# random_prt = {node: prt for node, prt in zip(G.nodes(), np.random.randint(0,high=max(unique_partitions),size=num_partitions))}
edge_proximities = {}
partition_sums = torch.zeros(num_partitions, word_vectors.shape[1])
partition_counts = torch.zeros(num_partitions)


other_copy = random_prt.copy()
for n1 in G.nodes():
    n2 =  (n2voc[n1], vocab, word_vectors, 2)[0]
    other_copy[n1]=random_prt[vocab[n2]]

for node, prt in random_prt.items():
    partition_sums[prt] += word_vectors[node]
    partition_counts[prt] += 1

# for n1 in G.nodes():
#     n2 = most_similar(n2voc[n1], vocab, word_vectors, 2)[0]
#     print("")
#     print(n1)
#     print(vocab[n2])
#     random_prt[n1]=random_prt[vocab[n2]]
    # tmp_G.add_edge(n1, vocab[n2])
statistics = []

for i in range(25):
    rollier = deque(maxlen=10)
    # rollier.append(1)
    rolling_movements = []
    candidates = []
    absolute_movements = []
    current_communities = np.unique(list(random_prt.values()))
    random_node_order = np.random.permutation(list(random_prt.keys()))
    for node_idx in random_node_order:
        # print("")
        curr_prt = random_prt[node_idx]
        # print("")
        # print(f"----{node_idx}----")
        store = None
        prt_candidates = set(
            random_prt[adj_node] 
            for adj_node 
            in list(G[node_idx]) 
            # if random_prt[adj_node] != curr_prt
            )
        empty_community = next(iter(set(range(min(current_communities), max(current_communities)+2)) - set(current_communities)))
        prt_candidates.add(empty_community)
        if len(prt_candidates) == 0:
            print(f"No candidates")
            continue
        # print(f"{len(prt_candidates)} candidates")
        prt_nodes = [
            node 
            for node, community 
            in random_prt.items() 
            if community == curr_prt 
            and node != node_idx
        ]
        # print(prt_nodes)
        curr_avg_diff = compute_avg_diff_to_centroid(prt_nodes + [node_idx], word_vectors) 
        curr_avg_diff_with_change = compute_avg_diff_to_centroid(prt_nodes, word_vectors) 
        # print(f"Node {node_idx} curr partition {curr_prt}: {curr_avg_diff} -> {curr_avg_diff_with_change}")
        giver_normalizer = len(prt_nodes)
        giver_normalizer = 1
        giver_gain = (curr_avg_diff_with_change - curr_avg_diff).cpu().numpy()/giver_normalizer if len(prt_nodes) != 0 else 0
        # if giver_gain < 0:
        #     # print(f"Giver gain is negative")
        #     continue
        # print(giver_gain, curr_avg_diff_with_change, curr_avg_diff)
        change_candidates = []
        for idx, prt_candidate in enumerate(prt_candidates):
            
            prt_candidate_nodes = [node for node, community in random_prt.items() if community == prt_candidate]
            candidate_avg_diff = compute_avg_diff_to_centroid(prt_candidate_nodes, word_vectors)
            candidate_avg_diff_with_change = compute_avg_diff_to_centroid(prt_candidate_nodes + [node_idx], word_vectors)
            receiver_gain = (candidate_avg_diff - candidate_avg_diff_with_change).cpu().numpy()
            # print(f"Node {node_idx} to partition {prt_candidate}: {change_score:.8f} = {candidate_avg_diff:.8f} - {candidate_avg_diff_with_change:.8f}")
            receiver_normalizer = len(prt_candidate_nodes)
            change_candidates.append((prt_candidate, ((receiver_gain*giver_gain)/receiver_normalizer), receiver_gain, giver_gain, receiver_normalizer))
                
        choose = 1
        maximum_gain = max(change_candidates, key=operator.itemgetter(choose))

        abs_gain = maximum_gain[1]
        receiver_gain = maximum_gain[2]
        giver_gain = maximum_gain[3]
        normalizer = maximum_gain[4]

        random_prt[node_idx] = maximum_gain[0]
        rollier.append(abs_gain)
        rolling_mean = np.mean(rollier)

        # rolling_movements.append(rolling_mean)
        # candidates.append(len(prt_candidates))
        # absolute_movements.append(maximum_gain[1].cpu().numpy())
        data_point = {
            "rol":rolling_mean, 
            "receiver_gain":receiver_gain, 
            "giver_gain":giver_gain, 
            "abs": abs_gain, 
            "candidates":len(prt_candidates),
            "partitions":len(set(random_prt.values()))
        }
        statistics.append(data_point)
    print(f"Final number of partitions is {len(set(random_prt.values()))}")
    # if rolling_mean < 0.04:
    #     break

data = pd.DataFrame(statistics)

fig, ax = plt.subplots(7,1)
fig.set_size_inches(10, 25)
visualize_benchmark_graph(G, pos, random_prt, ax=ax[0])
ax[1].plot(data["giver_gain"])
ax[2].plot(data["receiver_gain"])
ax[3].plot(data["abs"])
ax[4].plot(data["rol"])
ax[5].plot(data["candidates"])
ax[6].plot(data["partitions"])
ax[1].set_title("giver_gain")
ax[2].set_title("receiver_gain")
ax[3].set_title("abs")
ax[4].set_title("rol")
ax[5].set_title("candidates")
ax[6].set_title("partitions")
plt.tight_layout()


 # %%

def show_subset(prt_id, partition, G, pos, ax=None):
    subset = {node: 1 if prt == prt_id else -1  for node, prt in random_prt.items()}
    # print(G.nodes())
    return visualize_benchmark_graph(G, pos, partition=subset, ax=ax)

def show_all_identified_partitions(random_prt, G, pos):
    final_unique_partitions = set(random_prt.values())
    final_num = len(final_unique_partitions)
    fig, ax  = plt.subplots(final_num, 1)
    fig.set_size_inches(5, 5 * final_num)
    for prt_id, ax in zip(final_unique_partitions, ax):
        ax.set_title(f"Partition {prt_id}")
        show_subset(prt_id, random_prt, G, pos, ax)
    return plt.tight_layout()

show_all_identified_partitions(random_prt, G, pos)


# %%
