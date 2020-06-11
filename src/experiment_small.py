# %%
import numpy as np
import multiprocess as mp
import helper.utils as utils
import helper.quality_functions as qfunctions
import helper.visualization as viz
from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.glove_louvain import GloveMaximizationAlgorithm
import community as community_louvain
from networkx.algorithms import community as algorithms
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# %%
if __name__ == "__main__":
    G, pos = utils.generate_benchmark_graph(250, 0.1)
    pos = nx.spring_layout(G)
    algorithm = GloveMaximizationAlgorithm(fitness_function=None, verbose=True)
    partition = algorithm.run(G)
    # partition = community_louvain.best_partition(G)
    # %%
    viz.visualize_benchmark_graph(G, pos, partition)
    plt.show()
    # %%
    viz.show_all_identified_partitions(G, pos, partition)
    plt.show()
    # pickle.dump(algorithm, "last_run.pkl")


# %%
