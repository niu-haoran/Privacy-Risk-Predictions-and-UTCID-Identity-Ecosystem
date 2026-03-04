import numpy as np
import networkx as nx
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import argparse


def get_node_PII_explanation_dict(graph):
	node_PII_attribute_list = list(graph.nodes)
	node_PII_attribute_explanation_list = []

	for i in range(len(node_PII_attribute_list)):
		PII_attribute = node_PII_attribute_list[i]
		words = PII_attribute.split(" ")
		explanations = []
		for w in words:
			synsets = wn.synsets(w)
			for synset in synsets:
				explanations.append(synset.definition())
		node_PII_attribute_explanation_list.append(" ".join(explanations))
	node_PIIAttribute_explanation_dict = dict(zip(node_PII_attribute_list, node_PII_attribute_explanation_list))
	return node_PIIAttribute_explanation_dict


def find_padding_length(graph):
	node_PIIAttribute_explanation_dict = get_node_PII_explanation_dict(graph)
	model_name = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(model_name)
	node_context_embedding_length = {}
	for key, value in node_PIIAttribute_explanation_dict.items():
		node_context_embedding_length[key] = len(tokenizer(value, return_tensors = 'pt')["input_ids"].detach().numpy()[0])
	return np.median(list(node_context_embedding_length.values()))

def get_node_context_embeddings_index_dict(graph, max_length):
	node_PIIAttribute_explanation_dict = get_node_PII_explanation_dict(graph)
	model_name = "bert-base-uncased"
	tokenizer = BertTokenizer.from_pretrained(model_name)
	nodes = list(graph.nodes)
	index_context_dict = {}
	for index in range(len(nodes)):
		node = nodes[index]
		explain = node_PIIAttribute_explanation_dict[node]
		index_context_dict[index] = tokenizer(explain, return_tensors = 'pt',
			padding = 'max_length', max_length = max_length, truncation = True)['input_ids'].squeeze(0)
	return index_context_dict

class SeeGCN(torch.nn.Module):
	def __init__(self, num_features, token_id_sequence_length_median):
		super(SeeGCN, self).__init__() 
		self.conv1 = SAGEConv(num_features, 64, aggr = "mean")
		self.conv2 = SAGEConv(64, 16, aggr = "mean")
		self.fc00 = torch.nn.Linear(token_id_sequence_length_median*2, 128)
		self.context_norm = torch.nn.LayerNorm(128)
		self.fc01 = torch.nn.Linear(128, 64)
		self.fc02 = torch.nn.Linear(64, 32)
		self.fc03 = torch.nn.Linear(32, 8)
		self.fc04 = torch.nn.Linear(8, 1)
		self.fc1 = torch.nn.Linear(16, 8)
		self.fc2 = torch.nn.Linear(8, 1)

		self.fc = torch.nn.Linear(3, 1)

		self.fc11 = torch.nn.Linear(24, 16)
		self.fc12 = torch.nn.Linear(16, 8)
		self.fc13 = torch.nn.Linear(8, 1)
		self.sigmoid = torch.nn.Sigmoid()

	def encode(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = self.conv2(x, edge_index)
		return x

	def embedding_mlp(self, edge_embedding0, edge_embedding1, context_embedding0, context_embedding1):
		edge_agg = edge_embedding0*edge_embedding1
		edge_agg_16 = edge_agg.clone()
		edge_agg = self.fc1(edge_agg)
		edge_agg = self.fc2(edge_agg)
		context_embeddings = torch.cat((context_embedding0, context_embedding1), -1)
		context_embeddings = (context_embeddings - 15261.0)/15261.0
		context_forwards = self.fc00(context_embeddings.float())
		context_forwards = self.context_norm(context_forwards)
		context_forwards = self.fc01(context_forwards)
		context_forwards = self.fc02(context_forwards)
		context_forwards = self.fc03(context_forwards)
		fix2 = context_forwards.shape[0]
		context_forward_8 = context_forwards.clone().view(fix2, -1)
		context_forwards = self.fc04(context_forwards)
		fix = context_forwards.shape[0]
		context_forwards = context_forwards.view(fix, -1)

		ec_agg = torch.cat((edge_agg_16, context_forward_8), -1)
		ec_agg = self.fc11(ec_agg)
		ec_agg = self.fc12(ec_agg)
		ec_agg = self.fc13 (ec_agg)

		final_agg = torch.cat((edge_agg, context_forwards, ec_agg), -1)
		final_agg = self.fc(final_agg)
		return self.sigmoid(final_agg)

def train_link_predictor_3(model, train_data, val_data, index_context_dict, lr = 0.001, n_epochs = 100):
	optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
	criterion = torch.nn.BCELoss()
	aucs = []
	accs = []
	for epoch in range(1, n_epochs + 1):
		model.train()
		optimizer.zero_grad()
		embeddings = model.encode(train_data)
		negative_edge_index = negative_sampling(edge_index = train_data.edge_index, num_nodes = train_data.x.shape[0],
			num_neg_samples = train_data.edge_label_index.shape[1], method = 'sparse')
		edge_labels = torch.cat([torch.tensor([1]*train_data.edge_label_index.shape[1]), torch.tensor([0]*negative_edge_index.shape[1])])
		edge_embedding0 = torch.stack([embeddings[i] for i in train_data.edge_label_index[0]])
		edge_embedding1 = torch.stack([embeddings[i] for i in train_data.edge_label_index[1]])
		edge_embedding0_neg = torch.stack([embeddings[i] for i in negative_edge_index[0]])
		edge_embedding1_neg = torch.stack([embeddings[i] for i in negative_edge_index[1]])
		edge_embedding0 = torch.cat([edge_embedding0, edge_embedding0_neg])
		edge_embedding1 = torch.cat([edge_embedding1, edge_embedding1_neg])
		context_embedding0 = torch.stack([index_context_dict[int(edge_index)] for edge_index in train_data.edge_label_index[0]])
		context_embedding1 = torch.stack([index_context_dict[int(edge_index)] for edge_index in train_data.edge_label_index[1]])
		context_embedding0_neg = torch.stack([index_context_dict[int(edge_index)] for edge_index in negative_edge_index[0]])
		context_embedding1_neg = torch.stack([index_context_dict[int(edge_index)] for edge_index in negative_edge_index[1]])
		context_embedding0 = torch.cat([context_embedding0, context_embedding0_neg])
		context_embedding1 = torch.cat([context_embedding1, context_embedding1_neg])
		out = model.embedding_mlp(edge_embedding0, edge_embedding1, context_embedding0, context_embedding1).view(-1)
		loss = criterion(out.float(), edge_labels.float())
		loss.backward()
		optimizer.step()
		val_auc_score, val_loss, acc = eval_link_predictor_3(model, val_data, index_context_dict, criterion)
		aucs.append(round(val_auc_score, 2))
		accs.append(round(acc, 2))
		print(f"Epoch: {epoch:03d}, Train loss: {loss: .2f}, Val loss: {val_loss: .2f}, Val Acc: {acc:.2f}, Val Roc Auc Score: {val_auc_score:.2f}")
	print("The best auc score among " + str(n_epochs) + " epochs training is: "+str(np.max(aucs)))
	print("The best acc score among " + str(n_epochs) + " epochs training is: "+str(np.max(accs)))
	return model

def eval_link_predictor_3(model, val_data, index_context_dict, criterion):
	model.eval()
	embeddings = model.encode(val_data)
	edge_labels = val_data.edge_label
	edge_embedding0 = torch.stack([embeddings[i] for i in val_data.edge_label_index[0]])
	edge_embedding1 = torch.stack([embeddings[i] for i in val_data.edge_label_index[1]])
	context_embedding0 = torch.stack([index_context_dict[int(edge_index)] for edge_index in val_data.edge_label_index[0]])
	context_embedding1 = torch.stack([index_context_dict[int(edge_index)] for edge_index in val_data.edge_label_index[1]])
	out = model.embedding_mlp(edge_embedding0, edge_embedding1, context_embedding0, context_embedding1).view(-1)
	loss = criterion(out.float(), edge_labels.float())
	labels = edge_labels.detach().numpy()
	output = out.detach().numpy()
	pred = (output > 0.5).astype(int)
	correct = (pred == labels).sum()
	acc = correct/len(labels)
	return roc_auc_score(labels, output), loss, acc


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "train SeeGCN model")
	parser.add_argument("--graph_file", type = str, default = "utcid_identity_eco_graph.graphml", help = "indicating the name of the graph file")
	parser.add_argument("--train_data", type = str, default = "train_data.pt", help = "indicating the name of the train data file")
	parser.add_argument("--val_data", type = str, default = "val_data.pt", help = "indicating the name of the validation data file")
	parser.add_argument("--num_epochs", type = int, default = 10, help = "indicating the number of training epochs")
	args = parser.parse_args()
	G = nx.read_graphml(args.graph_file)
	train_data = torch.load(args.train_data)
	val_data = torch.load(args.val_data)
	num_epochs = args.num_epochs

	token_id_sequence_length_median = int(find_padding_length(G))
	model = SeeGCN(4, token_id_sequence_length_median)
	index_context_dict = get_node_context_embeddings_index_dict(G, token_id_sequence_length_median)
	model = train_link_predictor_3(model, train_data, val_data, index_context_dict,
		lr = 0.001, n_epochs = num_epochs)




