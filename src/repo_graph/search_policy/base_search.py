import torch
from typing import List, Dict

class SimilarSearchPolicy:
	""" Given a graph, and their embeddings, edges -> Retrieve the top k similar graphs to the query -> Expand the query graph to the top k similar graphs
	"""
	def __init__(self, embeddings: torch.Tensor, adj_matrix: torch.Tensor, type_matrices: torch.Tensor, index_to_name: Dict[int, str], search_config, expand_config) -> None:
		""" 

		Args:
			embeddings (torch.Tensor): a [num_nodes, embedding_dim] tensor
			edges (torch.Tesnsor): a [num_nodes, num_nodes] tensor denoting the edges between nodes, edge type agnostic
			type_matrices (torch.Tensor): a [num_edge_types, num_nodes, num_nodes] tensor
			index_to_name (Dict[int, str]): a dictionary mapping from index to the name of the node	
			search_config (dict): a dictionary containing the search config
			expand_config (dict): a dictionary containing the expand config
		"""
		if embeddings is None:
			raise ValueError("embeddings must be specified")
		if adj_matrix is None:
			raise ValueError("adj_matrix must be specified")
		if type_matrices is None:
			raise ValueError("type_matrices must be specified")
		if index_to_name is None:
			raise ValueError("index_to_name must be specified")

		assert embeddings.shape[0] == adj_matrix.shape[0]

		self.embeddings = embeddings
		self.adj_matrix = adj_matrix
		self.type_matrices = type_matrices
		self.index_to_name = index_to_name
		self.search_config = search_config
		self.expand_config = expand_config
		self.num_edge_types = self.type_matrices.shape[0]
		self.num_nodes	= self.adj_matrix.shape[0]
    
	def search(self, query):
		"""
		Args:
			query (torch.Tensor): a fixed dimension, should be the same as the embeddings of each node in the graphs

		Returns:
			List[int]: a list of centers of the top k similar graphs to the query

		Raises:
			NotImplementedError: [description]
		"""
		raise NotImplementedError
    
	def expand(self, centers):
		"""
		Args:
			query (torch.Tensor): a fixed dimension, should be the same as the embeddings of each node in the graphs

		Returns:
			List[int]: a list of indices of the top k similar graphs to the query
		"""
		raise NotImplementedError

	def retrieve(self, query, external_nodes=None):
		""" Retrieve the top k similar graphs to the query
		Args:
			query (torch.Tensor): a fixed dimension, should be the same as the embeddings of each node in the graphs

		Returns:
			expanded_indices (List[int]): a list of indices of the expanded graph
			retrieved_graph (torch.Tensor): a [expanded_num_nodes, expanded_num_nodes] tensor denoting the edges between nodes, edge type agnostic of the expanded graph
			retrieved_embedding (torch.Tensor): a [expanded_num_nodes, embedding_dim] tensor of the expanded graph
			translated_index_to_name (Dict[int, str]): a dictionary mapping from index to the name of the node of the expanded graph, new indices

		"""
		centers = self.search(query)
		expanded_indices = self.expand(centers)
		if external_nodes is not None:
			expanded_indices = list(set(expanded_indices + external_nodes))
		retrieved_graph = self.adj_matrix[expanded_indices][:, expanded_indices]
		retrieved_edge_types = self.type_matrices[:, expanded_indices][:, :, expanded_indices]
		retrieved_embedding = self.embeddings[expanded_indices]
		expanded_index_to_name = {k: self.index_to_name[k] for k in expanded_indices}
		translated_index_to_name = {k: v for k, v in enumerate(expanded_index_to_name.values())}
		map_original_translated = {k: v for k, v in zip(expanded_indices, translated_index_to_name.keys())}
		return expanded_indices, retrieved_graph, retrieved_edge_types, retrieved_embedding, translated_index_to_name, map_original_translated
		