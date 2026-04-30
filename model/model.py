import datasets
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from model.util import (
    attach_linkgram_matrices,
    inject_linkgram_attention,
    prepare_linkgram_inputs,
)

from . import (
    DATASET_ID,
    DATASET_VERSION,
    DEVICE,
    MAX_DISTANCE,
    MAX_INPUT_LENGTH,
    MODEL_ID,
    TEST_SPLIT,
)

test_split = datasets.load_dataset(DATASET_ID, DATASET_VERSION, TEST_SPLIT)

"""
print("\r\n")
print("Dataset format: ")
print(test_split)
print("\r\n")

print("Example Article: ")
print("\r\n")
print(test_split[1000]["article"])
print("\r\n")
"""

test_article = test_split[1000]["article"]

"""
TOKENIZE(R)
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""
This function tokenizes the input text and also builds the two token level
matrices required by our custom attention mechanism.
"""
tokens, token_distance_matrix, token_link_type_matrix, link_type_to_id = prepare_linkgram_inputs(
    test_article,
    tokenizer,
    max_length=MAX_INPUT_LENGTH,
    max_distance=MAX_DISTANCE,
    device=DEVICE,
)

"""
MODEL
"""
model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float32,
)

"""
Inject our custom attention now that we know how many link types were found
in this example.
"""
inject_linkgram_attention(model, max(1, len(link_type_to_id)), MAX_DISTANCE)

"""
Run generation after attaching the two token level matrices so the
encoder can use them when computing attention.
"""
attach_linkgram_matrices(model, token_distance_matrix, token_link_type_matrix)

with torch.no_grad():
    outputs = model.generate(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
    )

for item in outputs:
    res = tokenizer.decode(item, skip_special_tokens=True)
    print(res)
