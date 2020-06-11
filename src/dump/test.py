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
from algorithms.map_equation_louvain import MapEquationMaximization
from algorithms.louvain_core import LouvainCoreAlgorithm
from utils import post_transform, compute_experiment, modularity_wrapper, map_equation_wrapper, coverage_wrapper
from experiment import run_experiment


def increment(x):
    return x + 1

def decrement(x):
    return x - 1

if __name__ == "__main__":
    # louvain_algorithm = MapEquationMaximization(fitness_function=modularity_wrapper, verbose=True, max_iter=20)
    infomap_algorithm = LouvainCoreAlgorithm(fitness_function=map_equation_wrapper, verbose=True, max_iter=20)
    # coverage_algorithm = LouvainCoreAlgorithm(fitness_function=coverage_wrapper, verbose=True, max_iter=20)

    name = f"test"
    algorithms_for_experiment = {
        infomap_algorithm.run:"Test"
    }

    collected_data = []
    iterations = list(range(1))
    node_sizes = [250]
    mus = np.arange(0.1, 0.2, 0.1)
    configuration_set = list(itertools.product(*[iterations, algorithms_for_experiment.items(), node_sizes, mus]))

    print(compute_experiment(configuration_set[0]))


