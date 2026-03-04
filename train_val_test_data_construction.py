import torch
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils.convert import from_networkx
import argparse

def set_graph_simple_node_features(graph):
	node_in_degree = dict(graph.in_degree())
	node_out_degree = dict(graph.out_degree())
	node_betweenness_centrality = nx.betweenness_centrality(graph)
	node_closeness_centrality = nx.closeness_centrality(graph)
	nx.set_node_attributes(graph, node_in_degree, "node_in_degree")
	nx.set_node_attributes(graph, node_out_degree, "node_out_degree")
	nx.set_node_attributes(graph, node_betweenness_centrality, "node_betweenness_centrality")
	nx.set_node_attributes(graph, node_closeness_centrality, "node_closeness_centrality")

def construct_pyG_data_from_simple_features(graph, group_node_attrs = ["node_in_degree",
	                                                                   "node_out_degree",
	                                                                   "node_betweenness_centrality",
	                                                                   "node_closeness_centrality"]):
	data = from_networkx(graph, group_node_attrs = group_node_attrs)
	return data

def get_train_test_val_data(data, val_size_ratio, test_size_ratio):
	split = T.RandomLinkSplit(num_val = val_size_ratio, num_test = test_size_ratio, is_undirected = False, add_negative_train_samples = False)
	train_data, val_data, test_data = split(data)
	return train_data, val_data, test_data

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "train and validation data construction")
	parser.add_argument("--graph_file", type = str, default = "utcid_identity_eco_graph.graphml", help = "indicating the name of the graph file")
	parser.add_argument("--val_size_ratio", type = float, default = 0.1, help = "split ratio for validation size")
	parser.add_argument("--test_size_ratio", type = float, default = 0, help = "split ratio for test size")
	args = parser.parse_args()
	G = nx.read_graphml(args.graph_file)
	set_graph_simple_node_features(G)
	data = construct_pyG_data_from_simple_features(G)
	val_size_ratio = args.val_size_ratio
	test_size_ratio = args.test_size_ratio
	train_data, val_data, test_data = get_train_test_val_data(data, val_size_ratio, test_size_ratio)
	torch.save(train_data, "train_data.pt")
	torch.save(val_data, "val_data.pt")
	torch.save(test_data, "test_data.pt")
