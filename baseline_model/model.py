"""

This is a baseline inference run of the BART Language Model.

Using facebook/bart-base (pre-trained, not fine-tuned for summarization).

Model was loaded via Hugging Face.

"""

import datasets
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
)

#Set to CUDA gpu if available, else -> CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
TEST_SPLIT = "test"

test_split = datasets.load_dataset(DATASET_ID, DATASET_VERSION, split=TEST_SPLIT)

print("\r\n")
print("Dataset format: ")
print(test_split)
print("\r\n")

print("Example Article: ")
print("\r\n")
print(test_split[1000]["article"])
print("\r\n")

test_article = test_split[1000]["article"]

"""
TOKENIZE(R)
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

tokens = tokenizer(test_article, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
tensor = torch.tensor(tokens["input_ids"].to(DEVICE))
# print(tokens)

"""
MODEL

Pretrained BART model.
"""
model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float32,
)

# ensure no gradients are calculated or applied (we are doing inference, not training)
outputs = model.generate(tensor)


# use the tokenizer to decode the generated output tokens into text
for item in outputs:
    res = tokenizer.decode(item)
    print(res)
#some debuggng stuff
#print(model.__class__.__name__)
#print(model.config.activation_function)
