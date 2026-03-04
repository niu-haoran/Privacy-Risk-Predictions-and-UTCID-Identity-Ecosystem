import pandas as pd
import re

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import inflect

import argparse

# Check if the word is noun
def is_noun(word):
    tokenized_word = word_tokenize(word)
    tagged_word = pos_tag(tokenized_word)
    if len(tagged_word) >= 1:
        return tagged_word[0][1] in ('NN', 'NNS', 'NNP', 'NNPS')
    else:
        return False

def get_dataframe(file_name):
    return pd.read_csv(file_name)

def data_cleaning_process(df):
    inputs = df["inputs"].tolist()
    outputs = df["outputs"].tolist()
    loss = df["lossAmount"].tolist()
    indices_to_remove_for_empty = []

    # In the real dataset, missing entries of inputs and outputs are parsed as float
    # type(x) == float is used to detect missing values
    for i in range(len(inputs)):
        if type(inputs[i]) == float:
            indices_to_remove_for_empty.append(i)
    for i in range(len(outputs)):
        if type(outputs[i]) == float:
            indices_to_remove_for_empty.append(i)
 
    loss = [str(number) for number in loss]
    for i in range(len(loss)):
        if loss[i] == "nan":
            indices_to_remove_for_empty.append(i)
    indices_to_remove_for_empty = list(set(indices_to_remove_for_empty))
    
    df = df.drop(index = indices_to_remove_for_empty)
    inputs = df["inputs"].tolist()
    outputs = df["outputs"].tolist()
    loss = df["lossAmount"].tolist()
    
    for i in range(len(inputs)):
        inp = inputs[i].split(",")
        indices_to_remove = []
        for j in range(len(inp)):
            if len(inp[j]) == 1:
                indices_to_remove.append(j)
        inp = [k for m, k in enumerate(inp) if m not in indices_to_remove]
        inputs[i] = ",".join(inp)
    for i in range(len(outputs)):
        outp = outputs[i].split(",")
        indices_to_remove = []
        for j in range(len(outp)):
            if len(outp[j]) == 1:
                indices_to_remove.append(j)
        outp = [k for m, k in enumerate(outp) if m not in indices_to_remove]
        outputs[i] = ",".join(outp)

    inputs = [s.lower() for s in inputs]
    outputs = [s.lower() for s in outputs]
    inputs = [re.sub(r"\(.*?\)", "", s) for s in inputs]
    outputs = [re.sub(r"\(.*?\)", "", s) for s in outputs]
    inputs = [re.sub(r"'s|'", "", s) for s in inputs]
    outputs = [re.sub(r"'s|'", "", s) for s in outputs]

    # Check noun or not, and simplifying the terms by removing the descriptive word
    # Correct the errors introduced in the data collection process
    # Unify the terms
    for i in range(len(inputs)):
        # The code below using .strip() to be safe
        # But for simplicity, using inp = inputs[i].split(",") works fine as well
        inp = [x.strip() for x in inputs[i].split(",") if x.strip() != ""]
        for j in range(len(inp)):
            ns = inp[j].split(" ")
            indices = []
            for k in range(len(ns)):
                if is_noun(ns[k]) == False:
                    if k != 0 and k!= len(ns) - 1:
                        if is_noun(ns[k-1]) == False or is_noun(ns[k+1]) == False:
                            indices.append(k)
                    elif k == 0:
                        try:
                            if is_noun(ns[k+1]) == False:
                                indices.append(k)
                        except:
                            print("noun check err")
                    else:
                        indices.append(k)
            ns = [l for m, l in enumerate(ns) if m not in indices]
            inp[j] = " ".join(ns)
            if inp[j] == "medical diagnosis":
                inp[j] = "diagnosis"
            if inp[j] == "voter identfication card":
                inp[j] = "voter identification card"
        inputs[i] = ",".join(inp)
        
    
    for i in range(len(outputs)):
        # The code below using .strip() to be safe
        # But for simplicity, using inp = inputs[i].split(",") works fine as well
        outp = [x.strip() for x in outputs[i].split(",") if x.strip() != ""]
        for j in range(len(outp)):
            ns = outp[j].split(" ")
            indices = []
            for k in range(len(ns)):
                if is_noun(ns[k]) == False:
                    if k != 0 and k!= len(ns) - 1:
                        if is_noun(ns[k-1]) == False or is_noun(ns[k+1]) == False:
                            indices.append(k)
                    elif k == 0:
                        try:
                            if is_noun(ns[k+1]) == False:
                                indices.append(k)
                        except:
                            print("noun check err")
                    else:
                        indices.append(k)
            ns = [l for m, l in enumerate(ns) if m not in indices]
            outp[j] = " ".join(ns)
            if outp[j] == "medical diagnosis":
                outp[j] = "diagnosis"
            if outp[j] == "voter identfication card":
                outp[j] = "voter identification card"
        outputs[i] = ",".join(outp)

    # Change plural to singular word for unifying
    p = inflect.engine()
    for i in range(len(inputs)):
        inp = [x.strip() for x in inputs[i].split(",") if x.strip() != ""]
        for j in range(len(inp)):
            ns = inp[j].split(" ")
            indices = []
            for k in range(len(ns)):
                if ns[k] != "":
                    if p.singular_noun(ns[k]):
                        if len(wn.synsets(p.singular_noun(ns[k]))) > 0 and ns[k] != "data":
                            ns[k] = p.singular_noun(ns[k])
                else:
                    indices.append(k)
            ns = [l for m, l in enumerate(ns) if m not in indices]
            inp[j] = " ".join(ns)
        inputs[i] = ",".join(inp)
    
    for i in range(len(outputs)):
        outp = [x.strip() for x in outputs[i].split(",") if x.strip() != ""]
        for j in range(len(outp)):
            ns = outp[j].split(" ")
            indices = []
            for k in range(len(ns)):
                if ns[k] != "":
                    if p.singular_noun(ns[k]):
                        if len(wn.synsets(p.singular_noun(ns[k]))) > 0 and ns[k] != "data":
                            ns[k] = p.singular_noun(ns[k])
                else:
                    indices.append(k)
            ns = [l for m, l in enumerate(ns) if m not in indices]
            outp[j] = " ".join(ns)
        outputs[i] = ",".join(outp)
        
    return inputs, outputs, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser("description = data processing algorithm")
    parser.add_argument("--file_name", type = str, default = "synthetic_ITAP_data", help = "indicating the file name for data processing (without .csv)")
    args = parser.parse_args()

    csv_file = args.file_name + ".csv"
    df = get_dataframe(csv_file)
    inputs, outputs, loss = data_cleaning_process(df)
    df = pd.DataFrame()
    df["inputs"] = inputs
    df["outputs"] = outputs
    df["lossAmount"] = loss
    df.to_csv("input_output_loss_cases_for_graph_construction.csv", index = False)
    print("The intermediate result is stored in the csv file input_output_loss_cases_for_graph_construction.csv")








