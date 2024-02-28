import os
import torch
import pickle 

from src.utils import load_data
from src.repo_graph.repo_to_graph import embed_code
from src.repo_graph.parse_source_code import get_node_text
from src.repo_graph.search_policy.knn_search import ProximitySearchRadius
from src.repo_graph.search_policy.knn_search import ProximitySearchPattern
from src.repo_graph.search_policy.knn_search import ImportExpandRadius
from tqdm import tqdm
from typing import Dict, Any
from tree_sitter import Language, Parser
import argparse

PY_LANGUAGE = Language('/datadrive05/huypn16/treesitter-build/python-java.so', 'python')
parser = Parser()
parser.set_language(PY_LANGUAGE)

language = "python"
task = "retrieval"
settings = ["cross_file_first"]

cross_file_first_hard = load_data(task, language, "cross_file_first")["train"]["hard"]
cross_file_first_easy = load_data(task, language, "cross_file_first")["train"]["easy"]
cross_file_first = cross_file_first_hard + cross_file_first_easy

def args_parse():
    parser = argparse.ArgumentParser(description='Expand Strategies and assign gold label.')
    parser.add_argument('--search_policy', type=str, default="knn_pattern", help='Search policy')
    parser.add_argument('--rsg_path', type=str, help='Repo graphs (RSG) folder path')
    return parser.parse_args()

def get_function_class_name(code):
    # parse function or class name from repobench context snippet
    if code.startswith("def"):
        type_ = "function"
        query = """
            (function_definition 
                name: (identifier) @function.def
                parameters: (parameters) @function.parameters
                body: (_) @function.body
            (#not-a-child-of class_definition))
        """
        query = PY_LANGUAGE.query(query)
    elif code.startswith("class"):
        type_ = "class"
        query = """
        (class_definition 
            name: (identifier) @class.def
            body: (_) @class.body
        )
        """
        query = PY_LANGUAGE.query(query)
    else:
        return None, None
    tree = parser.parse(bytes(code, "utf8"))
    captures = query.captures(tree.root_node)
    node, _ = captures[0]
    return get_node_text(node.start_byte, node.end_byte, code), type_
        
def get_all_samples_from_repo(repo_name, dataset):
    samples = []
    for sample in dataset:
        if repo_name == sample["repo_name"].split("/")[1]:
            samples.append(sample)
    return samples

def get_gold_index(index_to_name, gold_snippet, imports_statement=None, next_line=None):
    # finding the closest function/class name inside the extracted graph to the gold snippet
    gold_name, type_ = get_function_class_name(gold_snippet)
    if gold_name is None:
        return None
    
    for idx, name in index_to_name.items():
        if gold_name in name.split("."):
            return idx
    
    return None

def finding_import_indexes_in_code(code, index_to_name, import_indexes):
    # finding the closest function/class name inside the extracted graph to the gold snippet
    called_imported_indexes = []
    weighted_called_imported_indexes = []
    for idx, name in index_to_name.items():
        if idx in import_indexes:
            if name.split(".")[-1] in code:
                called_imported_indexes.append(idx)
    
    for idx in called_imported_indexes:
        weighted_called_imported_indexes.append(code.count(index_to_name[idx].split(".")[-1]))
    
    #normalize the weight
    if len(weighted_called_imported_indexes) > 0:
        weighted_called_imported_indexes = [weight / sum(weighted_called_imported_indexes) for weight in weighted_called_imported_indexes]
        
    return called_imported_indexes, weighted_called_imported_indexes
    
def finding_gold_name_in_graph(gold_name, index_to_name):
    # finding the closest function/class name inside the extracted graph to the gold snippet
    for idx, name in index_to_name.items():
        if gold_name in name.split("."):
            return True
    return False

def label_graphs(repo_name, search_policy: str, retrieve_config: Dict[str, Any]):
    """ For each repo: graph loads the embeddings, type edges tensors, edges, node types. Then, for each sample, we will epxand the graph by policy, then extract the local graph using the found indices. We will label the gold snippet index, the query embedding, the import indexes, the kcge context, the index to name mapping
    
    Args:
        repo_name (str): the name of the repo
    
    Returns:
        output_samples (List[Dict]): New annotation of retrieved graphs
    """
    with open(f"data/repobench/repos_graphs_unixcoder/{repo_name}.pkl", "rb") as f:
        graph = pickle.load(f)
    graph["type_edges_tensor"] = torch.stack(list(graph["type_edges_tensor"].values()), dim=0)
    graph["name_to_index"] = {v: k for k, v in graph["index_to_name"].items()}
    # print(graph["type_edges_tensor"].shape)
    assert search_policy in ["knn_radius", "knn_pattern", "import_radius", "cosine_radius", "cosine_pattern"]
    assert "search_config" in retrieve_config
    assert "expand_config" in retrieve_config
    if search_policy == "knn_radius":
        search = ProximitySearchRadius(
            embeddings=graph["embeddings"],
            adj_matrix=graph["edge_tensor"],
            type_matrices=graph["type_edges_tensor"],
            index_to_name=graph["index_to_name"],
            search_config=retrieve_config["search_config"],
            expand_config=retrieve_config["expand_config"]
        )
    elif search_policy == "knn_pattern":
        index_to_type = {}
        # load explicitly the type of each node
        search = ProximitySearchPattern(
            embeddings=graph["embeddings"],
            adj_matrix=graph["edge_tensor"],
            type_matrices=graph["type_edges_tensor"],
            index_to_node_type=graph["index_to_node_type"],
            index_to_name=graph["index_to_name"],
            search_config=retrieve_config["search_config"],
            expand_config=retrieve_config["expand_config"],
            metric="knn"
        )
    elif search_policy == "import_radius":
        search = ImportExpandRadius(
            embeddings=graph["embeddings"],
            adj_matrix=graph["edge_tensor"],
            type_matrices=graph["type_edges_tensor"],
            index_to_name=graph["index_to_name"],
            search_config=retrieve_config["search_config"],
            expand_config=retrieve_config["expand_config"]
        )
    elif search_policy == "cosine_radius":
        search = ProximitySearchRadius(
            embeddings=graph["embeddings"],
            adj_matrix=graph["edge_tensor"],
            type_matrices=graph["type_edges_tensor"],
            index_to_name=graph["index_to_name"],
            search_config=retrieve_config["search_config"],
            expand_config=retrieve_config["expand_config"],
            metric="cosine"
        )
    elif search_policy == "cosine_pattern":
        search = ProximitySearchPattern(
            embeddings=graph["embeddings"],
            adj_matrix=graph["edge_tensor"],
            type_matrices=graph["type_edges_tensor"],
            index_to_node_type=graph["index_to_node_type"],
            index_to_name=graph["index_to_name"],
            search_config=retrieve_config["search_config"],
            expand_config=retrieve_config["expand_config"],
            metric="cosine"
        )
    else:
        print("Invalid search policy, currently only supports knn_radius and knn_pattern")
        exit(0)
    
    samples = get_all_samples_from_repo(repo_name, cross_file_first)
    num_nodes = len(graph["embeddings"])
    output_samples = []
    miss = 0
    hit = 0
    graph_miss = 0
    found_cover_ratios = 0
    for sample_idx, sample in enumerate(samples):
        import_indexes = []
        # embedding the preceeding code snippet as the query
        query_embedding = embed_code(sample["code"])
        if search_policy == "import_radius":
            search.temp_import_indexes = []
            for idx in range(len(sample["context"])):
                import_idx = get_gold_index(graph["index_to_name"], sample["context"][idx])
                if import_idx is not None:
                    search.temp_import_indexes.append(import_idx)
        # Extracting the local graph using found indices with KNN search as well as call graph query expanding
        expanded_indices, retrieved_graph, retrieved_edge_types, retrieved_embedding, translated_index_to_name, map_original_translated = search.retrieve(query_embedding, external_nodes=[graph["name_to_index"][sample["file_path"].split("/")[-1]]])
        sample["embeddings"] = retrieved_embedding
        sample["edge_tensor"] = retrieved_graph
        sample["index_to_name"] = translated_index_to_name
        sample["type_edges_tensor"] = retrieved_edge_types

        
        ## add query as the last node
        sample["embeddings"] = torch.cat([sample["embeddings"], query_embedding], dim=0)
        sample["edge_tensor"] = torch.cat([sample["edge_tensor"], torch.zeros(1, sample["edge_tensor"].shape[1])], dim=0)
        sample["edge_tensor"] = torch.cat([sample["edge_tensor"], torch.zeros(sample["edge_tensor"].shape[0], 1)], dim=1)
        translated_current_file_index = map_original_translated[graph["name_to_index"][sample["file_path"].split("/")[-1]]]
        sample["edge_tensor"][translated_current_file_index, sample["edge_tensor"].shape[1]-1] = 1
        sample["type_edges_tensor"] = torch.cat([sample["type_edges_tensor"], torch.zeros(5, 1, sample["type_edges_tensor"].shape[2])], dim=1)
        sample["type_edges_tensor"] = torch.cat([sample["type_edges_tensor"], torch.zeros(5, sample["type_edges_tensor"].shape[1], 1)], dim=2)
        sample["type_edges_tensor"][3, translated_current_file_index, sample["type_edges_tensor"].shape[2]-1] = 1
        sample["index_to_node_type"] = {v: graph["index_to_node_type"][k] for k, v in map_original_translated.items()}
        sample["index_to_node_type"][len(sample["index_to_node_type"])] = 1
        
        # * labeling which node is the gold snippet
        gold_index = get_gold_index(translated_index_to_name, sample["context"][sample["gold_snippet_index"]], imports_statement=sample["import_statement"], next_line=sample["next_line"])
        gold_name, type_ = get_function_class_name(sample["context"][sample["gold_snippet_index"]])
        
        # cases: gold_name is not a function or class name (global variable). TODO: handle this case by parsing import statements then use node code to represent this case
        if gold_name is None:
            # if the gold snippet is not in the graph, we will not use this graph for training
            continue
        # ! Checking if the gold snippet is in the retrieved graph, and retrieving statisitcs
        found_cover_ratios += (len(expanded_indices) / num_nodes)
        if gold_index is None:
            miss += 1
            gold_name, type_ = get_function_class_name(sample["context"][sample["gold_snippet_index"]])
            gold_name_in_graph = finding_gold_name_in_graph(gold_name, graph["index_to_name"])
            if gold_name_in_graph == False:
                graph_miss +=1
            continue
        else:
            hit += 1
        
        sample["all_gold_indexes"] = []
        for idx, name in translated_index_to_name.items():
            if gold_name in name.split("."):
                sample["all_gold_indexes"].append(idx)
        
        sample["gold_index"] = gold_index
        sample["index_to_name"] = translated_index_to_name
        sample["query"] = query_embedding
        for idx in range(len(sample["context"])):
            import_idx = get_gold_index(translated_index_to_name, sample["context"][idx])
            if import_idx is not None:
                import_indexes.append(import_idx)
            
        sample["import_indexes"] = import_indexes
        called_imported_indexes, weighted_called_imported_indexes = finding_import_indexes_in_code(sample["code"], translated_index_to_name, import_indexes)
        sample["weighted_called_imported_indexes"] = weighted_called_imported_indexes
        sample["type_edges_tensor"][1, sample["type_edges_tensor"].shape[2]-1, called_imported_indexes] = 1
        sample["kcge_context"] = [graph["contexts"][idx] for idx in expanded_indices]
        output_samples.append(sample)
    return output_samples, hit, miss, graph_miss, found_cover_ratios / len(samples)

if __name__ == "__main__":
    ## ! Configurations
    retrieve_radius_config = {
        "search_config": {
            "k": 8,
        },
        "expand_config": {
            "radius": 4,
            "max_size": 250
        }
    }
    retrieve_pattern_config = {
        "search_config": {
            "k": 1,
        },
        "expand_config": {
            "path_depth": 4
        }
    }
    # search_policy = "knn_pattern"
    # search_policy = "knn_radius"
    # search_policy = "import_radius"
    # search_policy = "cosine_radius"
    args = args_parse()
    search_policy = args.search_policy
    repo_names = os.listdir(args.rsg_path)
    hit = 0
    miss = 0
    graph_miss = 0
    found_cover_ratios = 0
    index = 0
    for repo_name in tqdm(repo_names):
        try:
            samples, _hit, _miss, _graph_miss, _found_cover_ratios = label_graphs(repo_name.split(".")[0], search_policy, retrieve_radius_config)
            hit += _hit
            miss += _miss # leauge of legends, miss fortune
            graph_miss += _graph_miss
            found_cover_ratios += _found_cover_ratios
            index += 1
            if index % 10 == 0:
                print("recall overall: ", hit / (hit + miss))
                print("percentage of graphs that miss the gold snippet overall: ", graph_miss / (miss + 1e-6))
                print("found cover ratios: ", found_cover_ratios / (index+1))
        except Exception as e:
            print(e)
            continue
        with open(f"data/repobench/repos_graphs_labeled_link_with_called_imported_edges/{repo_name}_labeled.pkl", "wb") as f:
            pickle.dump(samples, f)
