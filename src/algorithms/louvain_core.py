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
from collections import OrderedDict, Counter, deque
import pandas as pd
import multiprocess as mp
import matplotlib.cm as cm
import community as community_louvain
import scipy
from random import random
import operator
import time
import sys

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