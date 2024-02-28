from transformers import T5ForConditionalGeneration, AutoTokenizer
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.models.unixcoder import UniXcoder

checkpoint = "Salesforce/codet5p-220m"
device = "cuda:0" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint, truncation=True, max_length=512)
# model = T5ForConditionalGeneration.from_pretrained(checkpoint)
# model = model.encoder.to(device)

model = UniXcoder("microsoft/unixcoder-base").to(device)


def gather_type_edges(type_edges, name_to_index, type_edges_list, edges, name):
    for type_edge, edge in zip(type_edges_list, edges):
        if name_to_index[name] not in type_edges[type_edge]:
            type_edges[type_edge][name_to_index[name]] = []
        type_edges[type_edge][name_to_index[name]].append(name_to_index[edge])

def embed_code_t5(code):
    inputs = tokenizer.encode(code, return_tensors="pt", truncation=True, max_length=511).to(device)
    inputs = torch.concat((torch.tensor(tokenizer.cls_token_id).to(device).reshape(inputs.shape[0], 1), inputs), dim=1)
    with torch.no_grad():
        embedding = model.forward(inputs)["last_hidden_state"][:, 0, :].detach().cpu()
    return embedding

def embed_code_unix(code):
    inputs = model.tokenize([code], mode="<encoder-only>", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        _, embedding = model(torch.tensor(inputs).to(device))
    return embedding.detach().cpu()

def embed_code_t5_dict(dict_code):
    # get the embedding of the first token of the code [CLS] token
    embeddings = []
    texts = [dict_code[key] for key in dict_code]
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=511, padding="max_length").to(device)
    inputs = torch.concat((torch.tensor(tokenizer.cls_token_id).to(device).repeat(inputs["input_ids"].shape[0], 1), inputs["input_ids"]), dim=1)
    with torch.no_grad():
        for batch in inputs.split(128):
            embedding = model.forward(batch)["last_hidden_state"][:, 0, :].detach().cpu()
            embeddings.extend(torch.chunk(embedding, embedding.shape[0], dim=0))
    dict_code = {key: embeddings[idx] for idx, key in enumerate(dict_code)}
    return dict_code

def embed_code_unix_dict(dict_code):
    # get the embedding of the first token of the code [CLS] token
    embeddings = []
    texts = [dict_code[key] for key in dict_code]
    inputs = model.tokenize(texts, mode="<encoder-only>", truncation=True, max_length=512, padding=True)
    inputs = torch.concat([torch.tensor(_input).reshape(1, -1) for _input in inputs], dim=0).to(device)
    with torch.no_grad():
        for batch in inputs.split(256):
            _, embedding = model(batch)
            embedding = embedding.detach().cpu()
            embeddings.extend(torch.chunk(embedding, embedding.shape[0], dim=0))
    dict_code = {key: embeddings[idx] for idx, key in enumerate(dict_code)}
    return dict_code

def embed_code(code, model="unixcoder"):
    if model == "t5":
        return embed_code_t5(code)
    elif model == "unixcoder":
        return embed_code_unix(code)

def embed_code_dict(dict_code, model="unixcoder"):
    if model == "t5":
        return embed_code_t5_dict(dict_code)
    elif model == "unixcoder":
        return embed_code_unix_dict(dict_code)

def edge_dict_to_adjacency_tensor(edges, num_nodes):
    adjacency_tensor = torch.zeros(num_nodes, num_nodes)
    for node in edges:
        for edge in edges[node]:
            adjacency_tensor[node][edge] = 1
    return adjacency_tensor

def load_contexts_then_embed(contexts_files):
    """ For all the contexts files, we will embed the code and create the edges for the graph

    Args:
        contexts_files (List[Dict[filename, Dict[
            functions, List[Dict[name, context, edges, type_edges]]],
            classes, List[Dict[name, context, edges, type_edges, methods]]],
            code, Dict[name, context, edges, type_edges],
        ]]]): [description]

    Returns:
        [type]: [description]
    """
    embeddings = {}
    contexts = {}
    index_to_name = {}
    index_to_node_type = {}
    edges = {}
    type_edges = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
    # ! 0: code -> function, class (import)
    # ! 1: class, function, method -> class, method, function (call)
    # ! 2: code -> function, method (call)
    # ! 3: code -> function, class (ownership)
    # ! 4: class -> method (ownership)
    index = 0 
    #embed code
    for file in contexts_files:
        for function in file["functions"]:
            embeddings[index] = function["context"]
            contexts[index] = function["context"]
            index_to_name[index] = function["name"]
            index_to_node_type[index] = 1
            index += 1
        for cls in file["classes"]:
            # add class name and code to the embeddings
            embeddings[index] = cls["context"]
            contexts[index] = cls["context"]
            index_to_name[index] = cls["name"]
            index_to_node_type[index] = 2
            index += 1
            for method in cls["methods"]:
                embeddings[index] = method["context"]
                contexts[index] = method["context"]
                index_to_name[index] = method["name"]
                index_to_node_type[index] = 1
                index += 1
        embeddings[index] = file["code"]["context"]
        contexts[index] = file["code"]["context"]
        index_to_name[index] = file["code"]["name"]
        index_to_node_type[index] = 0
        index += 1
    
    name_to_index = {v: k for k, v in index_to_name.items()}
    #creating edges
    for file in contexts_files:
        for function in file["functions"]:
            # add edges for the function
            edges[name_to_index[function["name"]]] = [name_to_index[edge] for edge in function["edges"]]
            gather_type_edges(type_edges, name_to_index, function["type_edges"], function["edges"], function["name"])
        for cls in file["classes"]:
            # add edges for the class
            edges[name_to_index[cls["name"]]] = [name_to_index[edge] for edge in cls["edges"]]
            gather_type_edges(type_edges, name_to_index, cls["type_edges"], cls["edges"], cls["name"])
            # add edges for the methods
            for method in cls["methods"]:
                edges[name_to_index[method["name"]]] = [name_to_index[edge] for edge in method["edges"]]
                gather_type_edges(type_edges, name_to_index, method["type_edges"], method["edges"], method["name"])
        # add edges for the code node
        edges[name_to_index[file["code"]["name"]]] = [name_to_index[edge] for edge in file["code"]["edges"]]
        gather_type_edges(type_edges, name_to_index, file["code"]["type_edges"], file["code"]["edges"], file["code"]["name"])
    
    # batching the embeddings
    # embeddings = embed_code_dict(embeddings)
    embeddings = embed_code_dict(embeddings)
    
    return embeddings, edges, type_edges, contexts, index_to_name, index_to_node_type
    # return index_to_node_type

def visualize_graph(adjancency_tensor):
    rows, cols = np.where(adjancency_tensor.numpy() == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=5, font_size=5, font_color="red")
    plt.savefig("graph.png")
