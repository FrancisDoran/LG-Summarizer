from datasets import load_dataset
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
"""
data = load_dataset(DATASET_ID, DATASET_VERSION, keep_in_memory=False, streaming=True)

print(data)

train_split = data["train"]
test_split = data["test"]
evaluate_split = data["validation"]

"""
Model
"""
model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float32,
)
