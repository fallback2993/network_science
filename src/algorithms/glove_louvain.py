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
from utils import extract_partition_map, extract_community_map

import pickle
import torch
from torch import FloatTensor, LongTensor
from typing import Dict, Callable, List
import numpy as np
import matplotlib.pyplot as plt

class GloveMaximization(LouvainCoreAlgorithm):
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

        p_a_i, q_out_i, q_out, p_circle_i, p_u, partition_map, community_map = map_equation_essentials(G, partition_map)
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

        
        while True:
            random_order = np.random.permutation(G.nodes()) 
            had_improvement = False
            for node in random_order:            
                pass





        resulting_map = {id2node[node]: id2comm[community] for node, community in partition_map_copy.items()}
        print(f"Number of changes {num_changes}")
        return resulting_map, num_changes

    def _random_walk(self, a, i, iters):
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
    
    def _sample_graph(self, G, partition_map):
        walk_length = self.walk_length
        markov_matrix = np.array(nx.google_matrix(G, alpha=1))
        nodes = G.nodes()
        vocab = {f"node_{node}":node for node in nodes}
        n2voc = {node:name for name, node in vocab.items()}
        starting_point = np.random.choice(nodes)
        walk = self._random_walk(markov_matrix, starting_point, walk_length)
        sliding_windows = np.vstack((walk,np.roll(walk, -1), np.roll(walk, -2))).astype(np.int)
        return sliding_windows

    def _create_coorccurence_matrix(self, sample):
        cooccurence_matrix = np.zeros_like(nx.adjacency_matrix(self.G))
        center_node_pos = int(sample.shape[0]/2)
        for pos in range(walk_length):
            left_word = sliding_windows[0, pos]
            center_word = sliding_windows[1, pos]
            right_word = sliding_windows[2, pos]
            cooccurence_matrix[center_word,left_word] += 1
            cooccurence_matrix[center_word,right_word] += 1

        return cooccurence_matrix
    
    def _train(self):
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

def loss_fn(X_weighted: FloatTensor, W: FloatTensor, W_context: FloatTensor, 
            B: FloatTensor, B_context: FloatTensor, 
            X: FloatTensor, device:str = "cpu") -> FloatTensor:
    hypothesis = (torch.mm(W, W_context.transpose(0,1))  + B + B_context).type(torch.DoubleTensor).to(device)
    target = torch.log(X)
    squared_loss = ((hypothesis - target)**2)
    temp = torch.mul(X_weighted, squared_loss)
    result = torch.sum(temp)
    return result