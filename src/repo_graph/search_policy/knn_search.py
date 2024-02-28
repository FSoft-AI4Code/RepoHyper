import json
import time
import torch
from collections import defaultdict
from src.repo_graph.search_policy.base_search import SimilarSearchPolicy

DEFAULT_SEARCH_CONFIG = {
	"k": 6
}

DEFAULT_RADIUS_EXPAND_CONFIG = {
	"radius": 5,
	"max_size": 300
}

DEFAULT_PATTERN_EXPAND_CONFIG = {
	"path_depth": 5
}

class ProximitySearch(SimilarSearchPolicy):
	def __init__(self, embeddings, adj_matrix, type_matrices, index_to_name, search_config, expand_config, metric="knn"):
		if "k" not in search_config:
			raise ValueError("k must be specified in the search config")
		super().__init__(embeddings, adj_matrix, type_matrices, index_to_name, search_config, expand_config)
		self.metric=metric

	def search(self, query):
		# Calculate the Euclidean distance between the query and each item in the embeddings
		k = self.search_config["k"]
		if self.metric == "knn":
			distances = torch.sqrt(torch.sum((self.embeddings - query) ** 2, dim=1))
			_, indices = torch.sort(distances, descending=False)
   
		elif self.metric == "cosine":
			distances = torch.nn.functional.cosine_similarity(self.embeddings, query, dim=1)
			# Sort the distances and get the indices
			_, indices = torch.sort(distances, descending=True)
		# Return the indices of the top k items
		return indices[:k].tolist()
 

class ProximitySearchRadius(ProximitySearch):
	""" KNN searchs for the centers, expand the graph by the centers, then expand by simple bfs.
	"""
	def __init__(self, embeddings, adj_matrix, type_matrices, index_to_name, search_config=DEFAULT_SEARCH_CONFIG, expand_config=DEFAULT_RADIUS_EXPAND_CONFIG, metric="knn"):
		super().__init__(embeddings, adj_matrix, type_matrices, index_to_name, search_config, expand_config, metric)
		self.sanity_check()

	def sanity_check(self):
		if "k" not in self.search_config:
			raise ValueError("k must be specified in the search config")
		if "radius" not in self.expand_config:
			raise ValueError("radius must be specified in the expand config")
		if "max_size" not in self.expand_config:
			raise ValueError("max_size must be specified in the expand config")
        
	def bfs(self, centers):
		expanded_indices = set()
		for center in centers:
			visited = set()
			# distance = dict(int, int)
			queue = [(center, 0)]
			while queue:
				node, depth = queue.pop(0)
				if len(visited) >= self.expand_config["max_size"]:
					break
				if node not in visited:
					visited.add(node)
					if depth < self.expand_config["radius"]:
						coming_out, coming_in = [], []
						if self.adj_matrix[node].nonzero().numel() > 0:
							coming_out = self.adj_matrix[node].nonzero().squeeze(-1).tolist()
						if self.adj_matrix[:, node].nonzero().numel() > 0:
							coming_in = self.adj_matrix[:, node].nonzero().squeeze(-1).tolist()
						neighbors = coming_out + coming_in
						if len(neighbors) > 0:
							neighbors = set(neighbors)
							queue.extend([(neighbor, depth + 1) for neighbor in neighbors])
			# print(len(visited))
			for u in visited:
				expanded_indices.add(u)
		return list(expanded_indices)
	
	def expand(self, centers):
		""" Retrieve the neighbors of centers up to a radius, and return the adjacency matrix of the expanded graph

		Args:
			centers (List): a list of indices of the centers

		"""
		expanded_indices = self.bfs(centers)
		return expanded_indices

	def expanded_stat(self):
		raise NotImplementedError

class ImportExpandRadius(ProximitySearchRadius):
    def __init__(self, embeddings, adj_matrix, type_matrices, index_to_name, search_config=DEFAULT_SEARCH_CONFIG, expand_config=DEFAULT_RADIUS_EXPAND_CONFIG):
        super().__init__(embeddings, adj_matrix, type_matrices, index_to_name, search_config, expand_config)
        self.temp_import_indexes = []
    
    def search(self, query):
        return self.temp_import_indexes
  
class ProximitySearchPattern(ProximitySearch):
	""" KNN searchs for the centers, expand the graph by the centers, then search for the patterns.

	Args:
		ProximitySearch ([type]): [description]
	"""
	def __init__(self, embeddings, adj_matrix, type_matrices, index_to_node_type, index_to_name, search_config, expand_config, metric):
		self.index_to_node_type = index_to_node_type
		with open("src/repo_graph/search_policy/patterns.json", "r") as f:
			self.pattern = json.load(f)
		super().__init__(embeddings, adj_matrix, type_matrices, index_to_name, search_config, expand_config, metric)
		self.sanity_check()
	
	def sanity_check(self):
		if "path_depth" not in self.expand_config:
			raise ValueError("path_depth must be specified in the expand config")
 
	def find_all_path(self, source, destination):
		paths = []
		path_index = 0
		visited = set()
		path_buffer = []
		path_dict = {}
		self.dfs(source, None, visited, path_buffer, path_index, paths, None, path_dict)
		return path_dict
	
	def dfs(self, source, destination, visited, path_buffer, path_index, paths, in_edge_type, path_dict):
		""" Find all paths from source to destination

		Args:
			source: source vertex
			destination: destination vertex
			visited: contains the visited vertices
			path_buffer: maintaining the current path
			path_index: maintaining the current index in the path
			paths: maintaining the list of all paths
			in_edge_type (List): the type(s) of the edge from the previous vertex to the current vertex
		"""
		visited.add(source)
		# print(path_buffer)
		# print(path_index)
		# print(source)
		if path_index < len(path_buffer):
			path_buffer[path_index] = (source, in_edge_type)
		else:
			# ! a path from center -> destination(this source), for each destination(this source) append the satisfied path
			path_buffer.append((source, in_edge_type))
		path_index += 1

		if path_index >= self.expand_config["path_depth"]:
			# do not expand further
			path_index -= 1
			visited.remove(source)
			return

		if source not in path_dict:
			path_dict[source] = []
		
		path_dict[source].append(path_buffer[:path_index])
		for neighbor in self.adj_matrix[source].nonzero().squeeze(-1).tolist():
			in_edge_type = []
			for edge_type in range(self.num_edge_types):
				if self.type_matrices[edge_type][source][neighbor] != 0:
					in_edge_type.append(edge_type)
			assert len(in_edge_type) > 0, "There must be at least one edge type from source to neighbor"
			if neighbor not in visited:
				self.dfs(neighbor, destination, visited, path_buffer, path_index, paths, in_edge_type, path_dict)
    
		for neighbor in self.adj_matrix[:, source].nonzero().squeeze(-1).tolist():
			in_edge_type = []
			for edge_type in range(self.num_edge_types):
				if self.type_matrices[edge_type][:, source][neighbor] != 0:
					in_edge_type.append(edge_type)
			assert len(in_edge_type) > 0, "There must be at least one edge type from neighbor to source"
			if neighbor not in visited:
				self.dfs(neighbor, destination, visited, path_buffer, path_index, paths, in_edge_type, path_dict)

		path_index -= 1
		visited.remove(source)
 
	def walking(self, center):
		walking_time = time.time()
		# O(n^2)
		# only flow ups the allowed edge type
		pattern_node = set()
		center_type = self.index_to_node_type[center]
		allowed_pattern = self.pattern[str(center_type)]
		num_pattern = 0
		time_iter = 0
		paths_dict = self.find_all_path(center, None)
		# print(paths_dict[29])
		for destination in range(len(self.adj_matrix)):
			if destination == center:
				continue
			# print("destination", destination)
			st = time.time()
			if destination not in paths_dict:
				continue
			paths = paths_dict[destination]
			en = time.time() - st
			time_iter += en
			# print(paths)
			# check pattern
			for path in paths:
				# print(path)
				# print([self.index_to_node_type[node] for node, edge_type in path])
				walk_edge_patterns = [""]
				# print(path)
				assert path[-1][0] == destination, "The last node in the path must be the destination"
				for i in range(len(path) - 1):
					temp_patterns = []
					for current_pattern in walk_edge_patterns:
						for edge_type in path[i+1][1]: # many edge types for an incoming edge
							temp_patterns.append(current_pattern + str(edge_type))
					walk_patterns = temp_patterns # all the patterns from the source to the current node
				expand_to = False # whether to expand to the destination node, and all the nodes on the path to the destination as well
				num_pattern += len(walk_patterns)
				for pattern in walk_patterns:
					if pattern in allowed_pattern:
						expand_to = True
						break
				if expand_to:
					pattern_node.update([node for node, edge_types in path])
		# print("Time for expanding for each center", time.time() - walking_time)
		# print("Time for walking: ", time_iter)
		# print("For each center, there is ", num_pattern, "patterns")
		# print("For each center, there is ", len(list(pattern_node)), "expanded nodes")
		return list(pattern_node)
 
	def expand(self, centers):
		""" Expand the center, walking on its neighbours and find the patterns
  		"""
		expanded_indices = set()
		for center in centers:
			expanded_tree = self.walking(center) # list of indices in the dfs spanning tree around center
			expanded_indices.update(expanded_tree)
		return list(expanded_indices)