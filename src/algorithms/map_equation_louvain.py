from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.map_equation import map_equation_essentials
import numpy as np


class MapEquationMaximization(LouvainCoreAlgorithm):
    def initialize(self, G):
        initial_partition_map = dict(enumerate(G.nodes()))
        self.levels = []
        self.stats = {"local_moving": []}
        self.levels.append(initial_partition_map)
        # initial_fitness = self.fitness_function(initial_partition_map, G)
        # self.null_fitness.append(initial_fitness)
        self.level_fitness.append(0)
        self.level_graphs.append(G)
        self.gain_stats = []
        return G, initial_partition_map

    def local_movement(self, G, partition_map):
        def compute_weighted_entropy(probs, normalizer):
            return np.nansum((probs / normalizer) * np.log2(probs / normalizer))

        def compute_partition_probability(prt_id, node_weights, partition_map):
            community = [node for node, prt in partition_map.items() if prt == prt_id]
            return node_weights[community].sum()

        p_a_i, q_out_i, q_out, p_circle_i, p_u, partition_map, community_map = map_equation_essentials(G, partition_map)


        while True:
            random_order = np.random.permutation(G.nodes())
            had_improvement = False
            for node in random_order:
                for adj in G[node]:
                    tmp_q_a_i = p_a_i.copy() # partition_probabilities
                    giving_partition_probability = compute_partition_probability(partition_map[node], p_u, partition_map) - p_u[node]
                    receiving_partition_probability = compute_partition_probability(partition_map[node], p_u, partition_map) + p_u[node]
                    # old_

        # resulting_map = {id2node[node]: id2comm[community] for node, community in partition_map_copy.items()}
        # print(f"Number of changes {num_changes}")
        # return resulting_map, num_changes
