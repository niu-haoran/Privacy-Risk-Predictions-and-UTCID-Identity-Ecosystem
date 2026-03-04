import numpy as np
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
import argparse

class featureGCN(torch.nn.Module):
	def __init__(self, num_features):
		super(featureGCN, self).__init__()
		self.conv1 = SAGEConv(num_features, 64, aggr = "mean")
		self.conv2 = SAGEConv(64, 16, aggr = "mean")
		self.fc1 = torch.nn.Linear(16, 8)
		self.fc2 = torch.nn.Linear(8, 1)
		self.sigmoid = torch.nn.Sigmoid()
	def encode(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = self.conv2(x, edge_index)
		return x
	def embedding_mlp(self, edge_embedding0, edge_embedding1):
		edge_agg = edge_embedding0 * edge_embedding1
		edge_agg = self.fc1(edge_agg)
		edge_agg = self.fc2(edge_agg)
		return self.sigmoid(edge_agg)

def train_link_predictor_2(model, train_data, val_data, lr = 0.001, n_epochs = 100):
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
		out = model.embedding_mlp(edge_embedding0, edge_embedding1).view(-1)
		loss = criterion(out.float(), edge_labels.float())
		loss.backward()
		optimizer.step()
		val_auc_score, val_loss, acc = eval_link_predictor_2(model, val_data, criterion)
		aucs.append(round(val_auc_score, 2))
		accs.append(round(acc, 2))
		print(f"Epoch: {epoch:03d}, Train loss: {loss:.2f}, Val loss: {val_loss: .2f}, Val Acc:{acc:.2f}, Val Roc Auc Score:{val_auc_score:.2f}")
	print("The best auc score among "+str(n_epochs)+" training is: "+str(np.max(aucs)))
	print("The best acc score among "+str(n_epochs)+" training is: "+str(np.max(accs)))
	return model

def eval_link_predictor_2(model, val_data, criterion):
	model.eval()
	embeddings = model.encode(val_data)
	edge_labels = val_data.edge_label
	edge_embedding0 = torch.stack([embeddings[i] for i in val_data.edge_label_index[0]])
	edge_embedding1 = torch.stack([embeddings[i] for i in val_data.edge_label_index[1]])
	out = model.embedding_mlp(edge_embedding0, edge_embedding1).view(-1)
	loss = criterion(out.float(), edge_labels.float())
	labels = edge_labels.detach().numpy()
	output = out.detach().numpy()
	pred = (output > 0.5).astype(int)
	correct = (pred == labels).sum()
	acc = correct/len(labels)
	return roc_auc_score(labels, output), loss, acc

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "train FeatureGCN model")
	parser.add_argument("--train_data", type = str, default = "train_data.pt", help = "indicating the name of the train data file")
	parser.add_argument("--val_data", type = str, default = "val_data.pt", help = "indicating the name of the val data file")
	parser.add_argument("--num_epochs", type = int, default = 10, help = "indicating the number of the training epochs")
	args = parser.parse_args()
	train_data = torch.load(args.train_data)
	val_data = torch.load(args.val_data)
	num_epochs = args.num_epochs
	model = featureGCN(4)
	model = train_link_predictor_2(model, train_data, val_data, lr = 0.001, n_epochs = num_epochs)


