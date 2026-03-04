import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import argparse


def get_index_node_features_dict(graph):
	index_node_features_dict = {}
	nodes = list(graph.nodes)
	node_in_degree = dict(graph.in_degree())
	node_out_degree = dict(graph.out_degree())
	node_betweenness_centrality = nx.betweenness_centrality(graph)
	node_closeness_centrality = nx.closeness_centrality(graph)
	for index in range(len(nodes)):
		tmp = [node_in_degree[nodes[index]], node_out_degree[nodes[index]], 
		       node_betweenness_centrality[nodes[index]], node_closeness_centrality[nodes[index]]]
		index_node_features_dict[index] = torch.tensor(tmp).view(-1)
	return index_node_features_dict

class featureMLP(torch.nn.Module):
	def __init__(self, num_features):
		super(featureMLP, self).__init__()
		self.fc1 = torch.nn.Linear(num_features*2, 16)
		self.fc2 = torch.nn.Linear(16, 8)
		self.fc3 = torch.nn.Linear(8,1)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, edge_features0, edge_features1):
		feature_embedding = torch.cat((edge_features0, edge_features1), -1)
		feature_embedding = self.fc1(feature_embedding)
		feature_embedding = self.fc2(feature_embedding)
		feature_embedding = self.fc3(feature_embedding)
		return self.sigmoid(feature_embedding)


def train_link_predictor_1(model, train_data, val_data, graph, lr = 0.001, n_epochs = 100):
	optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
	criterion = torch.nn.BCELoss()
	features_index_dict = get_index_node_features_dict(graph) # can done outside of training function and pass as a parameter
	aucs = []
	accs = []
	for epoch in range(1, n_epochs + 1):
		model.train()
		optimizer.zero_grad()
		negative_edge_index = negative_sampling(edge_index = train_data.edge_index, 
			num_nodes = train_data.x.shape[0], 
			num_neg_samples = train_data.edge_label_index.shape[1], method = 'sparse')
		edge_labels = torch.cat([torch.tensor([1]*train_data.edge_label_index.shape[1]), torch.tensor([0]*negative_edge_index.shape[1])])
		edge_features0 = torch.stack([features_index_dict[int(i)] for i in train_data.edge_label_index[0]])
		edge_features1 = torch.stack([features_index_dict[int(i)] for i in train_data.edge_label_index[1]])
		edge_features0_neg = torch.stack([features_index_dict[int(i)] for i in negative_edge_index[0]])
		edge_features1_neg = torch.stack([features_index_dict[int(i)] for i in negative_edge_index[1]])
		edge_features0 = torch.cat([edge_features0, edge_features0_neg])
		edge_features1 = torch.cat([edge_features1, edge_features1_neg])
		out = model(edge_features0, edge_features1).view(-1)
		loss = criterion(out.float(), edge_labels.float())
		loss.backward()
		optimizer.step()
		val_auc_score, val_loss, acc = eval_link_predictor_1(model, val_data, criterion, features_index_dict)
		aucs.append(round(val_auc_score, 2))
		accs.append(round(acc, 2))
		print(f"Epoch: {epoch: 03d}, Train loss: {loss: .2f}, Val loss: {val_loss:.2f}, Val Acc: {acc: .2f}, Val AUC ROC Score: {val_auc_score: .2f}")
	print("The best auc score among "+str(n_epochs)+" epochs training is: "+str(np.max(aucs)))
	print("The best acc score among "+str(n_epochs)+" epochs training is: "+str(np.max(accs)))
	return model

def eval_link_predictor_1(model, val_data, criterion, features_index_dict):
	model.eval()
	edge_labels = val_data.edge_label
	edge_features0 = torch.stack([features_index_dict[int(i)] for i in val_data.edge_label_index[0]])
	edge_features1 = torch.stack([features_index_dict[int(i)] for i in val_data.edge_label_index[1]])
	out = model(edge_features0, edge_features1).view(-1)
	loss = criterion(out.float(), edge_labels.float())
	labels = edge_labels.detach().numpy()
	output = out.detach().numpy()
	pred = (output > 0.5).astype(int)
	correct = (pred == labels).sum()
	acc = correct/len(labels)
	return roc_auc_score(labels, output), loss, acc

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "train FeatureMLP model")
	parser.add_argument("--graph_file", type = str, default = "utcid_identity_eco_graph.graphml", help = "indicating the name of the graph file")
	parser.add_argument("--train_data", type = str, default = "train_data.pt", help = "indicating the name of the train data file")
	parser.add_argument("--val_data", type = str, default = "val_data.pt", help = "indicating the name of the validation data file")
	parser.add_argument("--num_epochs", type = int, default = 10, help = "indicating the number of training epochs")
	args = parser.parse_args()
	G = nx.read_graphml(args.graph_file)
	train_data = torch.load(args.train_data)
	val_data = torch.load(args.val_data)
	num_epochs = args.num_epochs
	model = featureMLP(4)
	model = train_link_predictor_1(model, train_data, val_data, G, lr = 0.001, n_epochs = num_epochs)

