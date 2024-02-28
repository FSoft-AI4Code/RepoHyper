import networkx as nx
import torch
import numpy as np 

def random_sampling(subgraphs_nodes, rate=0.5, minimum_redundancy=0, num_nodes=None):
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    while True:
        selected = np.random.choice(num_nodes, int(num_nodes*rate), replace=False)
        node_selected_times = torch.bincount(subgraphs_nodes[1][check_values_in_set(subgraphs_nodes[0], selected)], minlength=num_nodes)
        if node_selected_times.min() >= minimum_redundancy:
            # rate += 0.1 # enlarge the sampling rate 
            break
    return selected, node_selected_times

# Approach 1: based on shortets path distance 
def shortest_path_sampling(edge_index, subgraphs_nodes, stride=2, minimum_redundancy=0, random_init=False, num_nodes=None):
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1
    G = nx.from_edgelist(edge_index.t().numpy())
    G.add_nodes_from(range(num_nodes))
    if random_init:
        farthest = np.random.choice(num_nodes) # here can also choose the one with highest degree
    else:
        subgraph_size = torch.bincount(subgraphs_nodes[0], minlength=num_nodes)
        farthest = subgraph_size.argmax().item()

    distance = np.ones(num_nodes)*1e10
    selected = []
    node_selected_times = torch.zeros(num_nodes)

    for i in range(num_nodes):
        selected.append(farthest)
        node_selected_times[subgraphs_nodes[1][subgraphs_nodes[0] == farthest]] += 1
        length_shortest_dict = nx.single_source_shortest_path_length(G, farthest)
        length_shortest = np.ones(num_nodes)*1e10
        length_shortest[list(length_shortest_dict.keys())] = list(length_shortest_dict.values())
        mask = length_shortest < distance
        distance[mask] = length_shortest[mask]
        
        if (distance.max() < stride) and (node_selected_times.min() >= minimum_redundancy): # stop criterion 
            break
        farthest = np.argmax(distance)
    return selected, node_selected_times

def check_values_in_set(x, set, approach=1):
    assert min(x.shape) > 0
    assert min(set.shape) > 0
    if approach == 0:
        mask = sum(x==i for i in set)
    else:
        mapper = torch.zeros(max(x.max()+1, set.max()+1), dtype=torch.bool)
        mapper[set] = True
        mask = mapper[x]
    return mask

############################################### use dense input ##################################################
### this part is hard to change
 
def min_set_cover_sampling(edge_index, subgraphs_nodes_mask, random_init=False, minimum_redundancy=2):

    num_nodes = subgraphs_nodes_mask.size(0)
    if random_init:
        selected = np.random.choice(num_nodes) 
    else:
        selected = subgraphs_nodes_mask.sum(-1).argmax().item() # choose the maximum subgraph size one to remove randomness

    node_selected_times = torch.zeros(num_nodes)
    selected_all = []

    for i in range(num_nodes):
        # selected_subgraphs[selected] = True
        selected_all.append(selected)
        node_selected_times[subgraphs_nodes_mask[selected]] += 1
        if node_selected_times.min() >= minimum_redundancy: # stop criterion 
            break
        # calculate how many unused nodes in each subgraph (greedy set cover)
        unused_nodes = ~ ((node_selected_times - node_selected_times.min()).bool())
        num_unused_nodes = (subgraphs_nodes_mask & unused_nodes).sum(-1)
        scores = num_unused_nodes
        scores[selected_all] = 0
        selected = np.argmax(scores).item()

    return selected_all, node_selected_times
