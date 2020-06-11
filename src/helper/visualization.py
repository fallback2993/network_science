import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import math


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
