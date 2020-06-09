from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.map_equation import map_equation_essentials
import numpy as np

class MapEquationMaximization(LouvainCoreAlgorithm):
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

    
