import networkx as nx
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
	file_path = "/datadrive05/huypn16/knn-transformers/data/repobench/test_repos_graphs_labeled/abidria-api.pkl_labeled.pkl"
	with open(file_path, "rb") as f:
		data = pickle.load(f)
	print(data[1].keys())
	# print(data[0]["repo_name"])
	# print(data[0]["file_path"])
	print(data[1]["context"])
	# print(data[0]["import_statement"])
	print(data[1]["code"])
	entire_line = data[1]["next_line"]
	print(entire_line)
	# print(data[0]["gold_snippet_index"]) # most similar snippet indx
	# print(data[0]["embeddings"].shape) # [num_nodes, 768]
	# print(data[0]["edge_tensor"]) # [num_nodes * num_nodes]
	adj_matrix = data[1]["edge_tensor"] # [num_nodes * num_nodes]
	gold_index = data[1]["gold_index"] # none
	index_dict = data[1]["index_to_name"] #{num_nodes: name_entity}
	print(index_dict[gold_index])
	adj_matrix = adj_matrix.cpu().numpy()
	num_nodes = adj_matrix.shape[0]
	G = nx.Graph()
	for i in range(num_nodes):
		for j in range(num_nodes):
			if adj_matrix[i][j] == 1:
				G.add_edge(index_dict[i], index_dict[j])
	pos = nx.spring_layout(G, seed=42, k=0.9)
	labels = nx.get_edge_attributes(G, 'label')
	plt.figure(figsize=(120, 100))
	nx.draw(G, pos, with_labels=True, font_size=50, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
	nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=40, label_pos=0.3, verticalalignment='baseline')
	plt.title('Knowledge Graph')
	plt.savefig('graph.png')
	print("ok")
	
	# print(data[0]["query"].shape) # [1 * 768]