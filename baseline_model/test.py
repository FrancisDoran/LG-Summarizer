
import datasets
import torch
import transformers
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

"""

DATASET

"""

DATASET_ID = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
TEST_SPLIT = "test"

data = datasets.load_dataset(DATASET_ID, DATASET_VERSION, split=TEST_SPLIT)
print(data)

test_article = data[1000]["article"]

#print(data)
#print(test_article)
"""

TOKENIZER

"""

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

tokens = tokenizer(test_article, truncation=True, max_length=512, return_tensors="pt")
#vectors = tokenizer.encode(test_article, truncation=True, max_length=512)
#decoded = tokenizer.decode(vectors)
#print(tokens)
#print(vectors)

"""

MODEL

"""

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", device_map="auto", torch_dtype=torch.float32)

tens = torch.tensor(tokens["input_ids"].to(device))

res = model.generate(tens)

"""
This decodes the model output and prints the resulting tokens.
"""

for item in res:
    res = tokenizer.decode(item)
    print(res)





