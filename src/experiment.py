import numpy as np
import multiprocess as mp
from helper.utils import post_transform, compute_experiment, save_data
from helper.quality_functions import modularity_wrapper, flake_odf_wrapper, map_equation_wrapper, coverage_wrapper, conductance_wrapper, random_wrapper
from algorithms.louvain_core import LouvainCoreAlgorithm
from algorithms.glove_louvain import GloveMaximizationAlgorithm
from algorithms.label_propagation_louvain import HierarchicalLabelPropagation
from algorithms.random_louvain import RandomPropagation
from networkx.algorithms import community as algorithms
import itertools


def run_experiment(algorithms_for_experiment, backup, cores=8):

    iterations = list(range(0, 10))
    node_sizes = [250, 500]
    mus = np.arange(0.1, 0.8, 0.1)
    configuration_set = itertools.product(*[iterations, algorithms_for_experiment.items(), node_sizes, mus])

    cpu_cnt = cores
    pool = mp.Pool(processes=cpu_cnt)
    print(f"Running experiments in parallel with {cpu_cnt} cpus")
    parallel_execution_data = pool.imap_unordered(compute_experiment, configuration_set)

    for result in parallel_execution_data:
        save_data(result, backup)
        save_data(result, result["method"])


if __name__ == "__main__":
    louvain_algorithm = LouvainCoreAlgorithm(fitness_function=modularity_wrapper, verbose=True, max_iter=20)
    infomap_algorithm = LouvainCoreAlgorithm(fitness_function=map_equation_wrapper, verbose=True, max_iter=20)
    coverage_algorithm = LouvainCoreAlgorithm(fitness_function=coverage_wrapper, verbose=True, max_iter=20)
    conductance_algorithm = LouvainCoreAlgorithm(fitness_function=conductance_wrapper, verbose=True, max_iter=20)
    _ = LouvainCoreAlgorithm(fitness_function=flake_odf_wrapper, verbose=True, max_iter=20)
    lblprob_algorithm = post_transform(algorithms.asyn_lpa_communities)
    glove_algorithm_normal = GloveMaximizationAlgorithm(fitness_function=None, verbose=False, max_iter=50, mode=-1)
    glove_algorithm_0 = GloveMaximizationAlgorithm(fitness_function=None, verbose=False, max_iter=50, mode=0)
    glove_algorithm_1 = GloveMaximizationAlgorithm(fitness_function=None, verbose=False, max_iter=50, mode=1)
    glove_algorithm_2 = GloveMaximizationAlgorithm(fitness_function=None, verbose=False, max_iter=50, mode=2)
    hlbl_algorithm = HierarchicalLabelPropagation(fitness_function=None, verbose=False, max_iter=50)
    random_algorithm = RandomPropagation(fitness_function=None, verbose=False, max_iter=50)

    algorithms_for_experiment = {
        # infomap_algorithm.run:"Map Equation",
        # coverage_algorithm.run:"Coverage Maximization",
        # conductance_algorithm.run: "Conductance Maximization",
        # lblprob_algorithm:"Labelpropagation Algorithm",
        # louvain_algorithm.run: "Louvain Algorithm",
        # glove_algorithm_normal.run:
        # "Glove Mode Default",
        # glove_algorithm_0.run:
        # "Glove Mode 0",
        # glove_algorithm_1.run:
        # "Glove Mode 1",
        # glove_algorithm_2.run:
        # "Glove Mode 2",
        hlbl_algorithm.run: "Hierarchical Labelpropagation Algorithm",
        random_algorithm.run: "Random Propagation Algorithm"
    }
    cores = 10
    run_experiment(algorithms_for_experiment, "backup", cores)
