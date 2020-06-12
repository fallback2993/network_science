import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import math
from helper.utils import normalized_mutual_information


def visualize_benchmark_graph(G, pos, partition=None, ax=None):
    if partition:
        prt2idx = {key: idx for idx, key in enumerate(set(partition.values()))}
        partition = {node: prt2idx[prt] for node, prt in partition.items()}
        cmap = cm.get_cmap('jet', max(set(partition.values())) + 1)
        nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()), ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=40, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
    return None


def show_subset(prt_id, partition, G, pos, ax=None):
    subset = {node: 1 if prt == prt_id else -1 for node, prt in partition.items()}
    return visualize_benchmark_graph(G, pos, partition=subset, ax=ax)


def show_all_identified_partitions(G, pos, partition):
    final_unique_partitions = set(partition.values())
    final_num = len(final_unique_partitions)
    # num_groups = len(grouped_by_algorithm)
    num_rows = int(math.ceil(final_num / 4))
    fig, axes = plt.subplots(num_rows, 4, sharex=True, sharey=True)
    fig.set_size_inches(10, 5 * num_rows)
    ax = axes.flatten()
    for prt_id, ax in zip(final_unique_partitions, ax):
        ax.set_title(f"Partition {prt_id}")
        show_subset(prt_id, partition, G, pos, ax)
    return plt.tight_layout()


def show_reduction(algorithm, tmp_G, true_map, iterations=5):
    init_G, partition_map = algorithm.initialize(G=tmp_G)
    # partition_map = dict(enumerate(tmp_G.nodes()))
    pos = nx.spring_layout(tmp_G)
    fig, ax = plt.subplots(iterations, 2)
    fig.set_size_inches(10, 10)
    # VERBOSE = True
    ax[0][0].set_title(f"Start: {normalized_mutual_information(partition_map, true_map)}", fontsize=10)
    ax[0][0].set_axis_off()
    visualize_benchmark_graph(tmp_G, pos, partition_map, ax=ax[0][0])

    for idx in range(1, iterations):
        partition_map, fitness = algorithm.local_movement(tmp_G, partition_map)
        # print(algorithm.stats["cooccurence_matrices"][-1])
        algorithm.levels.append(partition_map)
        tmp_G, partition_map = algorithm.reduce_network(tmp_G, partition_map)
        ax[idx][0].set_title("Reduced map", fontsize=10)
        ax[idx][0].set_axis_off()
        visualize_benchmark_graph(tmp_G, nx.spring_layout(pos), partition_map, ax=ax[idx][0])

        backtracked_map = algorithm.decode_partition_map(len(algorithm.levels) - 1)
        ax[idx][1].set_title(normalized_mutual_information(backtracked_map, true_map), fontsize=10)
        ax[idx][1].set_axis_off()
        visualize_benchmark_graph(init_G, pos, backtracked_map, ax=ax[idx][1])

    ax[0][1].set_title(f"End: {normalized_mutual_information(true_map, true_map)}", fontsize=10)
    ax[0][1].set_axis_off()
    visualize_benchmark_graph(init_G, pos, true_map, ax=ax[0][1])


def show_intermediate_results(algorithm, G, true_map):
    # partition_map = dict(enumerate(tmp_G.nodes()))
    number_of_reductions = len(algorithm.levels)
    pos = nx.spring_layout(G)

    if number_of_reductions <= 1:
        backtracked_map = algorithm.decode_partition_map(number_of_reductions)
        visualize_benchmark_graph(G, pos, backtracked_map)
        return

    fig, ax = plt.subplots(number_of_reductions, 1)
    fig.set_size_inches(5, 10 * number_of_reductions)
    for idx in range(0, number_of_reductions):
        backtracked_map = algorithm.decode_partition_map(idx)
        ax[idx].set_title(f"Level {idx}: {normalized_mutual_information(backtracked_map, true_map)}", fontsize=10)
        ax[idx].set_axis_off()
        visualize_benchmark_graph(G, pos, backtracked_map, ax=ax[idx])
    return


def draw_plots(data):
    aggregated_over_trials = data.groupby(['method', 'N', 'µ']).mean()
    grouped_by_algorithm = aggregated_over_trials.groupby(['method'])
    num_groups = len(grouped_by_algorithm)
    num_rows = int(math.ceil(num_groups / 2))
    tmp = None
    fig, axes = plt.subplots(num_rows, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 5 * num_rows)
    axes = axes.flatten()
    for idx, (algorithm_name, algorithm_data) in enumerate(grouped_by_algorithm):
        axes[idx].set_title(algorithm_name)
        tmp = algorithm_data.reset_index().pivot(index='µ', columns='N', values='NMI')
        tmp.plot(ax=axes[idx])
        axes[idx].set_ylabel("Normalized Mutual Information")

    if num_groups < len(axes):
        fig.delaxes(*axes[num_groups:])
    fig.set_tight_layout(True)
    return plt.show()
