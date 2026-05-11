import json
import os

import datasets
from datasets import load_dataset
from peft import PeftModel
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from model import DEVICE, MAX_DISTANCE, MAX_INPUT_LENGTH, MODEL_ID
from model.model_diagnostics.model_diagnostics import DiagnosticCapture
from model.model_diagnostics.plot import MetricDictionary, create_bar_graph
from model.training.data_collator import LinkGramDataCollator
from model.util import (
    attach_linkgram_matrices,
    inject_linkgram_attention,
    prepare_linkgram_inputs,
)

"""
Evaluation script

Script to load and test the final trained version of the model.
Can be used to collect more metrics apart from what is already collected
during training.
"""

"""
Load dataset (test split)
"""
data = load_dataset("abisee/cnn_dailymail", "3.0.0")
test_split = data["test"]

"""
Load the trained model checkpoint.
"""
TRAIN_DIR = './bart_linkgram_training/bart_linkgram_training'

def get_best_checkpoint(train_dir):
    # Try to find the best checkpoint from trainer_state.json in the likely last checkpoint folder
    state_path = os.path.join(train_dir, 'checkpoint-8973', 'trainer_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
            best_ckpt = state.get('best_model_checkpoint')
            if best_ckpt:
                ckpt_name = os.path.basename(best_ckpt)
                return os.path.join(train_dir, ckpt_name)
    return os.path.join(train_dir, 'checkpoint-8973')

TRAINED_MODEL = get_best_checkpoint(TRAIN_DIR)
print(f"Using model checkpoint: {TRAINED_MODEL}")

"""
Load the trained model
"""
print(f"Loading base model: {MODEL_ID}")
base_model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=DEVICE,
)

inject_linkgram_attention(base_model, 5000, MAX_DISTANCE)

print(f"Loading adapter from: {TRAINED_MODEL}")
model = PeftModel.from_pretrained(base_model, TRAINED_MODEL)
model.eval()

"""
Load the vanilla version of the model for comparison
"""
print(f"Loading vanilla baseline model: {MODEL_ID}")
baseline_model = BartForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map=DEVICE,
)
baseline_model.eval()

"""
Tokenizer
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

"""
DataCollator
"""
link_type_to_id = {}
data_collator = LinkGramDataCollator(tokenizer, MAX_INPUT_LENGTH, MAX_DISTANCE, link_type_to_id, device=DEVICE)
print("Models and DataCollator loaded successfully.")

example = test_split[1000]
print("\n" + "="*80)
print("Article:", example["article"])
print("="*80)

# Use the data_collator to prepare inputs
batch = data_collator([example])

def single_comparison_run(custom_model, baseline_model, batch, tokenizer):
    with torch.no_grad():
        attach_linkgram_matrices(
            base_model, 
            batch["token_distance_matrix"], 
            batch["token_link_type_matrix"]
        )

        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            max_length=MAX_INPUT_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

        generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        baseline_generated_ids = baseline_model.generate(
            input_ids=batch["input_ids"],
            max_length=MAX_INPUT_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
        baseline_summary = tokenizer.decode(baseline_generated_ids[0], skip_special_tokens=True)

        print("\n[Vanilla BART Summary]:")
        print(baseline_summary)

        print("\n[LinkGram-Enhanced Summary]:")
        print(generated_summary)

        print("\n[Reference Summary]:")
        print(example["highlights"])

        diag = DiagnosticCapture()
        baseline_model_rouge, custom_model_rouge = diag.rouge_two_model_comparison(
            reference_summary=example["highlights"],
            baseline_model_generated_summary=baseline_summary,
            custom_model_generated_summary=generated_summary
        )

        return (baseline_model_rouge, custom_model_rouge)

baseline_model_scores, custom_model_scores = single_comparison_run(model, baseline_model, batch, tokenizer)

MetricDictionary = MetricDictionary()

# easy refactor, just use for loops.
# leaving it like this for now so it's easier to visualize interface between DiagnosticCapture, MetricDictionary, and the script above
MetricDictionary.add_metric("baseline", "rouge-1", "precision", baseline_model_scores["rouge1"].precision)
MetricDictionary.add_metric("baseline", "rouge-1", "recall", baseline_model_scores["rouge1"].recall)
MetricDictionary.add_metric("baseline", "rouge-1", "f1", baseline_model_scores["rouge1"].fmeasure)

MetricDictionary.add_metric("baseline", "rouge-2", "precision", baseline_model_scores["rouge2"].precision)
MetricDictionary.add_metric("baseline", "rouge-2", "recall", baseline_model_scores["rouge2"].recall)
MetricDictionary.add_metric("baseline", "rouge-2", "f1", baseline_model_scores["rouge2"].fmeasure)

MetricDictionary.add_metric("baseline", "rouge-len", "precision", baseline_model_scores["rougeL"].precision)
MetricDictionary.add_metric("baseline", "rouge-len", "recall", baseline_model_scores["rougeL"].recall)
MetricDictionary.add_metric("baseline", "rouge-len", "f1", baseline_model_scores["rougeL"].fmeasure)


MetricDictionary.add_metric("custom", "rouge-1", "precision", custom_model_scores["rouge1"].precision)
MetricDictionary.add_metric("custom", "rouge-1", "recall", custom_model_scores["rouge1"].recall)
MetricDictionary.add_metric("custom", "rouge-1", "f1", custom_model_scores["rouge1"].fmeasure)

MetricDictionary.add_metric("custom", "rouge-2", "precision", custom_model_scores["rouge2"].precision)
MetricDictionary.add_metric("custom", "rouge-2", "recall", custom_model_scores["rouge2"].recall)
MetricDictionary.add_metric("custom", "rouge-2", "f1", custom_model_scores["rouge2"].fmeasure)

MetricDictionary.add_metric("custom", "rouge-len", "precision", custom_model_scores["rougeL"].precision)
MetricDictionary.add_metric("custom", "rouge-len", "recall", custom_model_scores["rougeL"].recall)
MetricDictionary.add_metric("custom", "rouge-len", "f1", custom_model_scores["rougeL"].fmeasure)

create_bar_graph(MetricDictionary.get())







