def map_equation_wrapper(G, partition):
    # adjacency_matrix = nx.adjacency_matrix(G)
    # left_eigs = scipy.linalg.eig(adjacency_matrix.A, left=True)
    # print(adjacency_matrix.A)
    # print(left_eigs[0])
    numerical_stabilizer = np.finfo(float).eps
    transition_matrix = nx.algorithms.google_matrix(G)
    initial_starting_point = np.ones((transition_matrix.shape[0],1))/transition_matrix.shape[0]
    A = transition_matrix.T
    eigen_vector = initial_starting_point
    for k in range(1000):
        normalizer = np.linalg.norm(A * eigen_vector)
        eigen_vector = A * eigen_vector / normalizer
        # print(f"{k}. iteration")
        # print(b_k)
        # print(normalizer)
    stationary_node_distribution = np.array(eigen_vector/sum(eigen_vector)).squeeze()
    # print(stationary_node_distribution)
    eigen_value = normalizer + numerical_stabilizer
    community_map = extract_community_map(partition)
    # print(community_map)

    adjacent_nodes_per_node = {
        node : (partition[node],[partition[adjacent_node] for adjacent_node in G[node]]) 
        for idx, module in enumerate(community_map) 
        for node in module
    }
    num_partitions = max(set(partition.values()))+1
    partition_exit_links = np.zeros(num_partitions)
    partition_links_sums = np.zeros(num_partitions)
    partition_probabilities = np.zeros(num_partitions)
    for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items():
        partition_exit_links[node_partition] += sum(node_partition != np.array(neighbor_partitions)) 
        partition_links_sums[node_partition] += len(neighbor_partitions)
        partition_probabilities[node_partition] += stationary_node_distribution[node]
    # tmp = {node:(sum([partition[node] != partition[neighbor] for neighbor in adj]))  for node, adj in adjacent_nodes_per_node.items()}
    # tmp2 = {len(adj) for node, adj in adjacent_nodes_per_node.items()}
    # partition_exit_prob = {}
    # for node, prob in tmp.items():
    #     partition_exit_prob[partition[node]] = partition_exit_prob.setdefault(partition[node], 0) + prob
    partition_exit_prob = np.nan_to_num(partition_exit_links/partition_links_sums)
    node_exit_prob = {node: sum(node_partition != np.array(neighbor_partitions)) for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items()}

    q_out = partition_exit_prob.sum()
    q_i_out = partition_exit_prob 
    p_i_in = partition_probabilities
    denominator = p_i_in + q_i_out
    p_i_out_sum_p_i = stationary_node_distribution

    H_Q = - (q_i_out).dot(np.nan_to_num(np.log2(q_i_out)))
    H_P_i = - np.nan_to_num(q_i_out) * np.nan_to_num(np.log2(q_i_out)) - np.nan_to_num(p_i_in) * np.nan_to_num(np.log2(p_i_in))

    print(q_out, H_Q)
    print(denominator)
    print(H_P_i)
    print(q_out * H_Q)
    L = q_out * H_Q + denominator.dot(H_P_i)
    L = np.asarray(L).flatten()[0]
    return L

def map_equation_wrapper(G, partition):
    # adjacency_matrix = nx.adjacency_matrix(G)
    # left_eigs = scipy.linalg.eig(adjacency_matrix.A, left=True)
    # print(adjacency_matrix.A)
    # print(left_eigs[0])
    numerical_stabilizer = np.finfo(float).eps
    transition_matrix = nx.algorithms.google_matrix(G)
    initial_starting_point = np.ones((transition_matrix.shape[0],1))/transition_matrix.shape[0]
    A = transition_matrix.T
    eigen_vector = initial_starting_point
    for k in range(1000):
        normalizer = np.linalg.norm(A * eigen_vector)
        eigen_vector = A * eigen_vector / normalizer
        # print(f"{k}. iteration")
        # print(b_k)
        # print(normalizer)
    stationary_node_distribution = np.array(eigen_vector/sum(eigen_vector)).squeeze()
    # print(stationary_node_distribution)
    eigen_value = normalizer + numerical_stabilizer
    community_map = extract_community_map(partition)
    # print(community_map)

    adjacent_nodes_per_node = {
        node : (partition[node],[partition[adjacent_node] for adjacent_node in G[node]]) 
        for idx, module in enumerate(community_map) 
        for node in module
    }
    num_partitions = max(set(partition.values()))+1
    partition_exit_links = np.zeros(num_partitions)
    partition_links_sums = np.zeros(num_partitions)
    partition_probabilities = np.zeros(num_partitions)
    for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items():
        partition_exit_links[node_partition] += sum(node_partition != np.array(neighbor_partitions)) 
        partition_links_sums[node_partition] += len(neighbor_partitions)
        partition_probabilities[node_partition] += stationary_node_distribution[node]
    # tmp = {node:(sum([partition[node] != partition[neighbor] for neighbor in adj]))  for node, adj in adjacent_nodes_per_node.items()}
    # tmp2 = {len(adj) for node, adj in adjacent_nodes_per_node.items()}
    # partition_exit_prob = {}
    # for node, prob in tmp.items():
    #     partition_exit_prob[partition[node]] = partition_exit_prob.setdefault(partition[node], 0) + prob
    partition_exit_prob = np.nan_to_num(partition_exit_links/partition_links_sums)
    node_exit_prob = {node: sum(node_partition != np.array(neighbor_partitions)) for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items()}
    # print("CM")
    # print(community_map)
    # print("")
    # print("APN")
    # print(adjacent_nodes_per_node)
    # print("")
    # print("partition_exit_links")
    # print(partition_exit_links)
    # print("")
    # print("partition_links_sums")
    # print(partition_links_sums)
    # print("")
    # print("partition exit probs")
    # print(partition_exit_prob)
    # print("")
    # print("partition probs")
    # print(partition_probabilities)

    # print(sum(stationary_node_distribution))

    w_out = partition_exit_prob.sum()
    w_i_out = partition_exit_prob 
    w_a = stationary_node_distribution
    w_i = partition_probabilities
    w_i_out_add_w_i = w_i_out + w_i 

    term1 = w_out * np.nan_to_num(np.log2(w_out))
    term2 = 2 * w_i_out.dot(np.nan_to_num(np.log2(w_i_out)))
    term3 = w_a.dot(np.nan_to_num(np.log2(w_a)))
    term4 = w_i_out_add_w_i.dot(np.nan_to_num(np.log2(w_i_out_add_w_i)))

    print(term1, term2, term3, term4)
    L = term1 - term2 - term3 + term4
    L = np.asarray(L).flatten()[0]
    return L

def map_equation_wrapper(G, partition):

    numerical_stabilizer = np.finfo(float).eps
    transition_matrix = nx.algorithms.google_matrix(G)
    initial_starting_point = np.ones((transition_matrix.shape[0],1))/transition_matrix.shape[0]
    A = transition_matrix.T
    eigen_vector = initial_starting_point
    for k in range(1000):
        normalizer = np.linalg.norm(A * eigen_vector)
        eigen_vector = A * eigen_vector / normalizer
    stationary_node_distribution = np.array(eigen_vector/sum(eigen_vector)).squeeze()
    eigen_value = normalizer + numerical_stabilizer
    community_map = extract_community_map(partition) # For some reason partition zero misses
    num_links = len(G.edges())
    # print(community_map)

    adjacent_nodes_per_node = {
        node : (partition[node],[partition[adjacent_node] for adjacent_node in G[node]]) 
        for idx, module in enumerate(community_map) 
        for node in module
    }
    node_relative_weight = {node: len(neighbor_partitions)/(2*num_links) for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items()}
    node_relative_weight = dict(sorted(node_relative_weight.items()))
    # print(node_relative_weight)

    num_partitions = max(set(partition.values()))+1
    partition_ex_links = np.zeros(num_partitions)
    partition_in_links = np.zeros(num_partitions)
    partition_probabilities = np.zeros(num_partitions)
    partition_relative_weights = np.zeros(num_partitions)
    partition_relative_weights_exiting = np.zeros(num_partitions)
    for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items():
        partition_ex_links[node_partition] += sum(node_partition != np.array(neighbor_partitions)) 
        partition_in_links[node_partition] += sum(node_partition == np.array(neighbor_partitions)) 
        partition_probabilities[node_partition] += stationary_node_distribution[node]
        partition_relative_weights[node_partition] += node_relative_weight[node]
        # partition_relative_weights_exiting[node_partition] += []

    
    partition_links = ((partition_in_links/2)+partition_ex_links)
    partition_exit_prob = np.nan_to_num(partition_ex_links/partition_links)
    # partition_ex_relative_weights = 

    print(pd.DataFrame([
            partition_ex_links,
            partition_in_links, 
            partition_in_links + partition_ex_links,
            partition_ex_links/(partition_in_links + partition_ex_links), 
            partition_exit_prob,
            partition_probabilities, 
            partition_relative_weights
            ], 
        index=[
            "partition_ex_links", 
            "partition_in_links", 
            "partition_total_links",
            "partition_ex_rel_to_partition_links",
            "partition_exit_prob" ,
            "partition_probabilities", 
            "partition_relative_weights"
            ]))
    # node_exit_prob = {node: sum(node_partition != np.array(neighbor_partitions)) for node, (node_partition, neighbor_partitions) in adjacent_nodes_per_node.items()}

    # # Alternative approach
    # w_a = stationary_node_distribution
    # w_i = partition_probabilities
    # w_i_out = partition_exit_prob 
    # w_out = partition_exit_prob.sum()
    # w_i_out_add_w_i = w_i_out + w_i 

    # Paper approach
    w_a = np.array(list(node_relative_weight.values()))
    w_i = partition_relative_weights
    w_i_out = np.nan_to_num(partition_ex_links/num_links)
    # w_i_out = np.nan_to_num(partition_ex_links/(partition_in_links + partition_ex_links))
    w_out = w_i_out.sum()
    w_i_out_add_w_i = w_i_out + w_i 

    term1 = w_out * np.nan_to_num(np.log2(w_out))
    term2 = 2 * w_i_out.dot(np.nan_to_num(np.log2(w_i_out)))
    term3 = w_a.dot(np.nan_to_num(np.log2(w_a)))
    term4 = w_i_out_add_w_i.dot(np.nan_to_num(np.log2(w_i_out_add_w_i)))

    print("")
    print(term1, term2, term3, term4)
    L = term1 - term2 - term3 + term4
    L = np.asarray(L).flatten()[0]
    return L