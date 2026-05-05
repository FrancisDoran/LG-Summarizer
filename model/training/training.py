import datasets
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from model import (
    DATASET_ID,
    DATASET_VERSION,
    DEVICE,
    MAX_DISTANCE,
    MAX_INPUT_LENGTH,
    MODEL_ID,
    TEST_SPLIT,
)
from model.util import (
    attach_linkgram_matrices,
    inject_linkgram_attention,
    prepare_linkgram_inputs,
)

# link type dictionary GLOBAL
link_type_to_id = {}

"""
Dataset

CNN DAILY NEWS format...

DatasetDict = {
    train: Dataset(),
    validation: Dataset(),
    test: Dataset()
}

Dataset = {
    features: [],
    num_rows: int
}

features = ['article', 'highlights', 'id']

where article is original article,
and highlights is reference, human-made, summary
"""
data = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0")

test_article = data[TEST_SPLIT][1000]["article"]
#print(test_article)

"""
Tokenizer
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""
Model
"""
model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float32,
)

tokens, token_distance_matrix, token_link_type_matrix, link_type_to_id = prepare_linkgram_inputs(
    test_article,
    tokenizer,
    max_length=MAX_INPUT_LENGTH,
    max_distance=MAX_DISTANCE,
    device=DEVICE,
)

inject_linkgram_attention(model, max(1, len(link_type_to_id)), MAX_DISTANCE)

attach_linkgram_matrices(model, token_distance_matrix, token_link_type_matrix)

"""
for layer in model.model.encoder.layers:
    print(layer)
"""

"""
PEFT Config
"""

modules_to_train = []
for layer in model.model.encoder.layers:
    modules_to_train.append(layer.self_attn.distance_bias)
    # distance bias layers
    #print(layer.self_attn.distance_bias)
    modules_to_train.append(layer.self_attn.link_type_bias)
    # link type bias layers
    #print(layer.self_attn.link_type_bias)

peft_config = LoraConfig(
    target_modules=modules_to_train,
    modules_to_save=["link_type_to_id"],
)

peft_model = get_peft_model(
        model=model,
        peft_config=peft_config,
)

peft_model.print_trainable_parameters()
