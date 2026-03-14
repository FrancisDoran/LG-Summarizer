import os

import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM as seq2seq

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = seq2seq.from_pretrained("facebook/bart-large-cnn")

model.to(device)

tokenizer.save_pretrained("./bart-large-cnn")
model.save_pretrained("./bart-large-cnn")
