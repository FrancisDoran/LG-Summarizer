from datasets import load_dataset
import numpy as np
from peft import LoraConfig

"""
Dataset
"""
data = load_dataset("abisee/cnn_dailymail", "3.0.0", keep_in_memory=False, streaming=True)

train_set = data["train"]

