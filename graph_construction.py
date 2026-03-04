from data_preprocessing import data_cleaning_process
import pandas as pd
import networkx as nx
import argparse

def graph_construction(inputs, outputs):
	G = nx.DiGraph()
	weight_dict = {}
	for i in range(len(inputs)):
		inp = inputs[i].split(",")
		outp = outputs[i].split(",")
		for j in range(len(inp)):
			for k in range(len(outp)):
				if (inp[j], outp[k]) not in weight_dict:
					weight_dict[(inp[j], outp[k])] = 1
				else:
					weight_dict[(inp[j], outp[k])] += 1
	for (input_attr, output_attr), weight in weight_dict.items():
		G.add_edge(input_attr, output_attr, weight = weight)
	return G


def graph_construct_subgraph_of_different_sizes(inputs, outputs, sample_size, random_state = None):
	df = pd.DataFrame()
	df["inputs"] = inputs
	df["outputs"] = outputs
	df = df.sample(sample_size, random_state = random_state)

	inputs = df["inputs"].tolist()
	outputs = df["outputs"].tolist()
	G = graph_construction(inputs, outputs)
	return G

def graph_construct_subgraph_of_different_loss_amount_threshold(df, thres):
	df = df[df["lossAmount"] > thres]
	df = df.reset_index(drop = True)
	inputs, outputs, _ = data_cleaning_process(df)
	print("Number of inputs and outputs (number of cases):")
	print((len(inputs), len(outputs)))
	G = graph_construction(inputs, outputs)
	return G

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = "Graph Construction")

	parser.add_argument("--graph_construction_type", type = str, default = "basic", help = "indicating if you would like to construct graphs (use 'basic'), construct subgraphs with specific sizes (use 'sub_w_size'), or construct subgraphs with a loss amount threshold (use 'sub_w_loss').")
	parser.add_argument("--file_name", type = str, default = "synthetic_ITAP_data.csv", help = "indicating an input file name")
	parser.add_argument("--sample_size", type = int, default = 100, help = "indicating a sample size if choosing to build a subgraph")
	parser.add_argument("--loss_threshold", type = float, default = 1000, help = "indicating a threshold of loss amount")

	args = parser.parse_args()

	if args.graph_construction_type == "basic":
		df = pd.read_csv(args.file_name)
		inputs = df["inputs"].tolist()
		outputs = df["outputs"].tolist()
		G = graph_construction(inputs, outputs)
		nx.write_graphml(G, "utcid_identity_eco_graph.graphml")
	elif args.graph_construction_type == "sub_w_size":
		df = pd.read_csv(args.file_name)
		inputs = df["inputs"].tolist()
		outputs = df["outputs"].tolist()
		sample_size = args.sample_size
		if sample_size > len(df):
			raise ValueError("Please choose a sample size <= "+str(len(df)))
		else:
			G = graph_construct_subgraph_of_different_sizes(inputs, outputs, sample_size)
			nx.write_graphml(G, "utcid_identity_eco_subgraph_w_size_"+str(sample_size)+".graphml")
	elif args.graph_construction_type == "sub_w_loss":
		df = pd.read_csv(args.file_name)
		thres = args.loss_threshold
		G = graph_construct_subgraph_of_different_loss_amount_threshold(df, thres)
		nx.write_graphml(G, "utcid_identity_eco_subgraph_w_loss_"+str(thres)+".graphml")
	else:
		raise ValueError("Unknown graph_construction_type input. Valid input: basic | sub_w_loss | sub_w_size")
