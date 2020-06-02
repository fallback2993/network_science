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

class LouvainAlgorithm:
    
    fitness_function = None
    levels = []
    level_fitness = [] 

    def __init__(
        self, 
        resolution=0.01, 
        max_iter=-1, 
        fitness_function=community_louvain.modularity, 
        verbose=False,
        improvements=False,
        max_local_movements=sys.maxsize
        # stop_after
        ):
        self.fitness_function = fitness_function
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_local_movements = max_local_movements
        self.resolution = resolution
        self.improvements = improvements
        super().__init__()
    
    def inititalize(self, G):
        initial_partition_map = dict(enumerate(self.G.nodes()))
        self.levels = []
        self.levels.append(initial_partition_map)
        return G, initial_partition_map


    def run_louvain(self, G):
        self.G = G
        _, initial_partition_map = self.inititalize(self.G)
        # self.level_fitness.append(-100)
        result = self.run_iteration(G, initial_partition_map, self.max_iter)
        if self.verbose:
            print(f"Final results are in! Algorithm found {len(np.unique(list(result.values())))} communities")
        backtracked_partitioning = self.decode_partition_map(len(self.levels)-1)
        return backtracked_partitioning

    def run_iteration(self, G, initial_partition_map, stop_after=-1):
        new_partition_map, final_fitness = self.local_movement(G, initial_partition_map)
        if stop_after == 0:
            print(F"STOP: User defined stop after {self.stop_after} iterations")
            return new_partition_map
        if new_partition_map == initial_partition_map:
            print(f"STOP: Both community_maps are the same -> {new_partition_map == initial_partition_map}")    
            return new_partition_map
        if len(self.level_fitness) and final_fitness - self.level_fitness[-1] < self.resolution:
            print(f"STOP: Gain of {final_fitness - self.level_fitness[-1]} fell below {self.resolution}")    
            return new_partition_map


        if self.improvements:
            clusters = self._extract_community_map()
        self.levels.append(new_partition_map)
        self.level_fitness.append(final_fitness)

        new_G, reduced_partitions = self.reduce_network(G, new_partition_map)
        
        # return new_partition_map, reduced_adjacency, reduced_partitions
        reduced_community_map = self._extract_community_map(new_partition_map)
        
        # print("GESFSDFDAS", new_G.nodes())
        # print("GESFSDFDAS", reduced_partitions)
        final_partition = self.run_iteration(new_G, reduced_partitions, stop_after-1)
            # partition_map_old = new_partition_map
            # for i in range(len(num_communities)):
            #     community_map[i] = reduced_community_map[i]
        return final_partition

    def local_movement(self, G, partition_map):
        partition_map_copy = partition_map.copy()
        partition_map_result = None
        initial_fitness = self.fitness_function(partition_map_copy, G)
        # initial_fitness = map_equation_wrapper(partition_map_copy, G)
        container_of_n_last_gains = deque(maxlen=10)
        container_of_n_last_gains.append(100)
        # fitness = -100
        cnt = 0
        has_improvement = True
        # while has_improvement:
        # try:
        #     np.random.permutation(G.nodes())
        # except Exception as identifier:
        #     pass
        last_improvement = np.absolute(initial_fitness)
        while True and len(G.nodes()) > 1:
        #     pass
            random_order = np.random.permutation(G.nodes()) 
            had_improvement = False
            for node in random_order:
                # node = np.random.choice(G.nodes()) if len(G.nodes()) > 1 else list(G.nodes())[0]
                # print(node)
                current_communities = np.unique(list(partition_map_copy.values()))
                empty_community = next(iter(set(range(min(current_communities), max(current_communities)+2)) - set(current_communities)))
                # print(empty_community)
                candidates = [partition_map_copy[adjacent_node] for adjacent_node in G[node]] + [empty_community]
                # print(candidates)
                gains = [self._compute_fitness(G, partition_map_copy, initial_fitness, node, candidate_community) for candidate_community in candidates] 
                maximum_gain = max(gains, key=operator.itemgetter(1))
                if maximum_gain[1] > 0:
                    rolling_average = np.mean(container_of_n_last_gains)

                    if rolling_average < self.resolution:
                        print(f"Rolling average of {rolling_average:.8f} is below {self.resolution}") 
                        break   
                    had_improvement = True
                    verbose_str2 = f"Increase {initial_fitness:.8f} -> {maximum_gain[2]:.8f}"
                    verbose_str3 = f"Moved node {maximum_gain[3]} to community {maximum_gain[4]}"
                    print(maximum_gain[1])
                    print(rolling_average)
                    partition_map_copy = maximum_gain[0]
                    last_improvement = maximum_gain[1]
                    # verbose_str1 = f"Gain {maximum_gain[1]:.8f} < {self.resolution}? {last_improvement < self.resolution}"
                    initial_fitness = maximum_gain[2]
                    container_of_n_last_gains.append(maximum_gain[1])
                    # cnt=0
                    if self.verbose: print(f"{verbose_str2} | {verbose_str3}\n")
                cnt += 1
            if had_improvement == False:
                print("No further improvement!")
                break
                # if last_improvement < self.resolution:
                #     break
            # else:
            #     last_improvement = 0
            #     cnt+=1

            # partition_map_copy[node] = curr_community   
            
            # print(f"New BIGLI {has_improvement} with {fitness} : {initial_fitness}")
            if cnt > self.max_local_movements:
                print(f"Max iteration reached!")
                break
            
                    # break
            # if cnt > self.max_iter: break         
        print(f"--- Local movement completed with {initial_fitness} ---\n")      
            

        return partition_map_copy, last_improvement

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
        ebunch = [key+({'weight':value},) for key, value in counter.most_common()] # TODO
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

    def _compute_fitness(self, G, partition, old_fitness, node, community):
        partition_copy = partition.copy()
        partition_copy[node] = community 
        fitness = self.fitness_function(partition_copy, G)
        gain = fitness - old_fitness
        return partition_copy, gain, fitness, node, community