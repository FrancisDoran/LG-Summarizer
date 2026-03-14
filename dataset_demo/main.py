import datasets
import numpy as np

data = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0")


print("\r\n")

train_size = len(data["train"])
print(f"Training set size: {train_size}")

validation_size = len(data["validation"])
print(f"Training set size: {validation_size}")

train_size = len(data["test"])
print(f"Training set size: {train_size}")

print("\r\n")

print("-" * 50)
print("Article example: ")
print("-" * 50)
print(data["validation"][0]['article'])
print("\r\n")

print("-" * 50)
print("Corresponding summary example: ")
print("-" * 50)
print(data["validation"][0]['highlights']) # get the highlights (summary) 
                                                # section from the first 
                                                # index of the validation set

print("-" * 50)

print("\r\n")



"""
This script loads one of our datasets and tokenizes the first entry in the dataset.

from datasets import load_dataset, get_dataset_split_names, Dataset
import torch
import nltk

# check if cuda is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if no split is specified, a dict is returned
dataset_base = load_dataset("cnn_dailymail", "3.0.0", split="train")

# download the punkt tokenizer models
nltk.download("punkt")

### parse sentences -> tokenize
sentences = nltk.sent_tokenize(dataset_base[0]["article"])

sentence_bag = list()
word_bag = list()
# iterate over each sentence...
for sentence in sentences:
    sentence_bag.append(sentence)
    # for each token in the sentence, add it to the token bag
    for word in sentence.split():
        word_bag.append(word)

def sentences():
    return sentence_bag

def words():
    return word_bag

print("\r\n")
print(f"Tokenized sentences: {sentence_bag}")
print("\r\n")

print(f"Tokenized words: {word_bag}")
print("\r\n")

def sentences():
    return sentence_bag

#make tensor
#tensor = dataset_base.with_format("torch", device=device)

#print(tensor[0])

training_data = dataset["train"]
testing_data = dataset["test"]

example = training_data[0]

print(f"{example}")
"""
