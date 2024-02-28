from src.repo_graph.repo_to_graph import load_contexts_then_embed, edge_dict_to_adjacency_tensor
from src.repo_graph.parse_source_code import parse_source
import os
import pickle
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import sys
import torch

REPOS_FOLDER = "data/repobench/repos/"

def sizeof(obj):
    return len(pickle.dumps(obj))

def repo_to_graph(repo_path, call_graph_json_path):
    contexts_files = parse_source(repo_path, call_graph_json_path)
    embeddings, edges, type_edges, contexts, index_to_name, index_to_node_type = load_contexts_then_embed(contexts_files)
    edge_tensor = edge_dict_to_adjacency_tensor(edges, len(embeddings))
    type_edges_tensor = {k: edge_dict_to_adjacency_tensor(type_edges[k], len(embeddings)) for k in type_edges}

    return embeddings, edge_tensor, type_edges_tensor, contexts, index_to_name, index_to_node_type

def main(repo_name):
        try:
            call_graph_json_path = os.path.join("data/repobench/repos_call_graphs", repo_name + ".json")
            with open(call_graph_json_path, "r") as f:
                data = json.load(f)
                if len(data) == 0:
                    pass
            embeddings, edge_tensor, type_edges_tensor, contexts, index_to_name, index_to_node_type = repo_to_graph(os.path.join(REPOS_FOLDER, repo_name), call_graph_json_path)
            with open(os.path.join("data/repobench/repos_graphs_unixcoder", repo_name + ".pkl"), "wb") as f:
                pickle.dump({
                    "embeddings": torch.concat(list(embeddings.values()), dim=0), 
                    "edge_tensor": edge_tensor, 
                    "type_edges_tensor": type_edges_tensor, 
                    "contexts": contexts, 
                    "index_to_name": index_to_name,
                    "index_to_node_type": index_to_node_type
                }, f)
        
        except Exception as e:
            print(e)
            pass    
def fix():
    repo_names = os.listdir(REPOS_FOLDER)
    for name in tqdm(repo_names[280:]):
        main(name)
if __name__ == "__main__":
    # main()
    fix()