if __name__ == "__main__":
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "examples"

    import networkx as nx
    from infomap import Infomap, MapEquation, InfoNode
    from algorithms.map_equation import map_equation, map_equation_improved
    from helper.utils import generate_benchmark_graph
    from helper.visualization import visualize_benchmark_graph
    import matplotlib.pyplot as plt

    # G = nx.karate_club_graph()
    G = nx.barbell_graph(4, 1)
    # G = nx.bull_graph()
    # G = nx.generators.erdos_renyi_graph(10, 0.5)
    # G = nx.generators.cubical_graph()
    # G = generator.planted_partition_graph(5,50, p_in=0.3, p_out=0.01)
    # G, pos = generate_benchmark_graph(500, 0.2)
    # G = tmp_G
    InfoNode()
    MapEquation()
    pos = nx.spring_layout(G)
    im = Infomap()

    # G = nx.Graph()
    # G.add_node(1)
    # G.add_node(2)
    # G.add_edge(1,2)
    for e in G.edges(data='weight', default=1):
        # print(e)
        im.addLink(e[0], e[1], e[2])
    im.run()

    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
    print("\n#node module")
    result = {node.node_id: node.module_id - 1 for node in im.tree if node.is_leaf}
    infomap_partition = dict(sorted(result.items()))
    # infomap_partition = dict(sorted(result.items()))

    codelength, index_codelength, module_codelength = map_equation(G, infomap_partition)
    print("")
    print("------------------")
    print("")
    codelength_, index_codelength_, module_codelength_ = map_equation_improved(G, infomap_partition, False)

    print("")
    print("Result")
    print(f"Calculated_1 {codelength} = {index_codelength} + {module_codelength}")
    print(f"Calculated_2 {codelength_} = {index_codelength_} + {module_codelength_}")
    print(f"Correct is   {im.codelengths[0]} = {im.index_codelength} + {im.module_codelength}")
    print(f"1. Difference is {im.codelengths[0]-codelength}")
    print(f"2. Difference is {im.codelengths[0]-codelength_}")
    visualize_benchmark_graph(G, pos, infomap_partition)
    # plt.show()