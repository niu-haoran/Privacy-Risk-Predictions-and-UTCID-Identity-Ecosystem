import pandas as pd
import random
import argparse

df = pd.read_csv("pii_attribute_list_for_synthetic_data_generation.csv")
pii_list = df['PII Attribute'].tolist()

def synthetic_data_generator(pii_list, number_of_cases_needed = 50, maxLossAmount = 50000.0, minLossAmount = 0.0):
	inputs = []
	outputs = []
	lossAmount = []
	df = pd.DataFrame()
	for i in range(number_of_cases_needed):
		input_item_number = random.randint(1, int(len(pii_list)/2))
		output_item_number = random.randint(1, int(len(pii_list)/2))
		inputs.append(",".join(random.sample(pii_list, input_item_number)))
		outputs.append(",".join(random.sample(pii_list, output_item_number)))
		lossAmount.append(random.uniform(minLossAmount, maxLossAmount))
	df["inputs"] = inputs
	df["outputs"] = outputs
	df["lossAmount"] = lossAmount
	return df

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description = 'Generate synthetic data which has the same format as the real ITAP data.')

	parser.add_argument('--number_of_cases_needed', type = int, default = 50, help = 'indicating how many number of cases you need to generate')
	parser.add_argument('--maxLossAmount', type = float, default = 50000.0, help = 'indicating the maximum loss amount you allow for each synthetic case')
	parser.add_argument('--minLossAmount', type = float, default = 0.0, help = 'indicating the minimum loss amount you allow for each synthetic case')

	args = parser.parse_args()

	df = synthetic_data_generator(pii_list, args.number_of_cases_needed, args.maxLossAmount, args.minLossAmount)
	df.to_csv("synthetic_ITAP_data.csv", index = False)
	print("The synthetic data is stored in the csv file named synthetic_ITAP_data.csv")
