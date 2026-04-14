import datasets
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from model.util import inject_linkgram_attention, prepare_linkgram_inputs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
TEST_SPLIT = "test"
MAX_INPUT_LENGTH = 512
MAX_DISTANCE = 10

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
Run generation while also passing in the two token level matrices so the
encoder can use them when computing attention.
"""
with torch.no_grad():
    outputs = model.generate(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        token_distance_matrix=token_distance_matrix,
        token_link_type_matrix=token_link_type_matrix,
    )

for item in outputs:
    res = tokenizer.decode(item, skip_special_tokens=True)
    print(res)
