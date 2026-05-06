import datasets
import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from model import (
    DATASET_ID,
    DATASET_VERSION,
    DEVICE,
    MAX_DISTANCE,
    MAX_INPUT_LENGTH,
    MODEL_ID,
    TEST_SPLIT,
)
from model.training.data_collator import LinkGramDataCollator
from model.util import inject_linkgram_attention, prepare_linkgram_inputs

# link type dictionary GLOBAL
link_type_to_id = {}

"""
Dataset
"""
data = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0")

train_split = data["train"]
test_split = data["test"]
evaluation_split = data["validation"]

"""
Tokenizer
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""
Model
"""
model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
)

link_type_to_id = {}

# Inject attention with some headroom for new link types found during training
inject_linkgram_attention(model, max(5000, len(link_type_to_id) + 1000), MAX_DISTANCE)

"""
PEFT Config
"""
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # these modules have a LoRA adapter injected into them
    #   this is necessary because the Embedding layers that hold the bias tensors
    #   use these for interfacing with the model.
    #
    #These are the only "unfrozen" parts of the original baseline model
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    # these modules are to be trained
    modules_to_save=["distance_bias", "link_type_bias"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

peft_model = get_peft_model(
    model=model,
    peft_config=peft_config,
)

peft_model.print_trainable_parameters()

"""
Data Collator

This dynamically calculates the linkgram tensors for each batch
instead of precomputing them for the whole dataset.
"""
data_collator = LinkGramDataCollator(
    tokenizer=tokenizer,
    max_length=MAX_INPUT_LENGTH,
    max_distance=MAX_DISTANCE,
    link_type_to_id=link_type_to_id,
)

"""
TrainingArguments
"""
training_args = TrainingArguments(
    output_dir="./bart_linkgram_training",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=10,
    load_best_model_at_end=True,
    # Do not change...this allows the link bias tensors to be passed throughout the whole model
    remove_unused_columns=False, 
)

"""
Trainer
"""
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=evaluation_split,
    data_collator=data_collator,
)

# To start training:
trainer.train()
