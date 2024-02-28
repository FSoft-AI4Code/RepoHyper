import torch
from torch_sparse import SparseTensor # for propagation

def k_hop_subgraph(edge_index, num_nodes, num_hops):
    # return k-hop subgraphs for all nodes in the graph
    row, col = edge_index
    sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes), trust_data=True)
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)] # each one contains <= i hop masks
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator==-1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N 
    node_mask = (hop_indicator >= 0) # N x N dense mask matrix
    return node_mask, hop_indicator


from torch_cluster import random_walk
def random_walk_subgraph(edge_index, num_nodes, walk_length, p=1, q=1, repeat=1, cal_hops=True, max_hops=10):
    """
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)  Setting it to a high value (> max(q, 1)) ensures 
            that we are less likely to sample an already visited node in the following two steps.
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
            if q > 1, the random walk is biased towards nodes close to node t.
            if q < 1, the walk is more inclined to visit nodes which are further away from the node t.
        p, q ∈ {0.25, 0.50, 1, 2, 4}.
        Typical values:
        Fix p and tune q 

        repeat: restart the random walk many times and combine together for the result

    """
    row, col = edge_index
    start = torch.arange(num_nodes, device=edge_index.device)
    walks = [random_walk(row, col, 
                         start=start, 
                         walk_length=walk_length,
                         p=p, q=q,
                         num_nodes=num_nodes) for _ in range(repeat)]
    walk = torch.cat(walks, dim=-1)
    node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    # print(walk.shape)
    node_mask.fill_(False)
    node_mask[start.repeat_interleave((walk_length+1)*repeat), walk.reshape(-1)] = True
    if cal_hops: # this is fast enough
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float())>0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator==-1) & next_mask] = i+1
            if hop_indicator[node_mask].min() != -1:
                break 
        return node_mask, hop_indicator
    return node_mask, None

from torch_sparse import mul 
def ppr_topk(edge_index, num_nodes, k=10, alpha=0.1, t=5):
    # this implementation is currently slow comparing to the other two,
    # the problem is the number of power, the matrix multiplication needs time. 
    ## another problem of using fixed k: different nodes will have same number of local nodes.
    ## this in general is not good. Ideally should depends on node degree?5
    """
        k: keep top k nodes left, this should include the orignal node 
        t: number of power iterations 
        alpha:restart probability (teleport probability)

        This function is more suitable for denser graph. For sparse graph, fixed k for all nodes 
        is a big issue. 
    """
    start = torch.eye(num_nodes, dtype=torch.float)
    sparse_adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
    # row normalize the sparse adj
    deg_inv = sparse_adj.sum(-1).pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0
    sparse_adj = mul(sparse_adj, deg_inv.view(-1, 1))
    ppr = start # each row is a probability
    for _ in range(t):
        ppr = (1-alpha)*sparse_adj.matmul(ppr) + alpha*start 
    _, node_idx = torch.topk(ppr, k, dim=-1)

    node_mask = node_idx.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    node_mask.fill_(False)
    node_mask[torch.arange(num_nodes).repeat_interleave(k), node_idx.reshape(-1)] = True
    return node_mask, None
