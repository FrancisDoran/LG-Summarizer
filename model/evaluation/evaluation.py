import json
import os

import datasets
from datasets import load_dataset
from peft import PeftModel
import torch
from tqdm import tqdm
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

def evaluate_models(test_data, num_examples=100):
    diag = DiagnosticCapture()
    
    metrics_to_track = ["rouge1", "rouge2", "rougeL"]
    stats = ["precision", "recall", "fmeasure"]

    # allocate "space" for adding up scores across examples
    accumulated_baseline = {metric: {stat: 0.0 for stat in stats} for metric in metrics_to_track}
    accumulated_custom = {metric: {stat: 0.0 for stat in stats} for metric in metrics_to_track}

    print(f"Starting evaluation on {num_examples} examples...")

    for i in tqdm(range(num_examples), desc="Evaluating"):
        example = test_data[i]
        batch = data_collator([example])
        
        with torch.no_grad():

            # Baseline model
            baseline_generated_ids = baseline_model.generate(
                input_ids=batch["input_ids"],
                max_length=MAX_INPUT_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
            baseline_summary = tokenizer.decode(baseline_generated_ids[0], skip_special_tokens=True)
            
            attach_linkgram_matrices(
                base_model, 
                batch["token_distance_matrix"], 
                batch["token_link_type_matrix"]
            )
            
            # Custom model
            custom_generated_ids = model.generate(
                input_ids=batch["input_ids"],
                max_length=MAX_INPUT_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
            custom_summary = tokenizer.decode(custom_generated_ids[0], skip_special_tokens=True)
            
            reference_summary = example["highlights"]
            
            # Get rouge scores
            baseline_scores = diag.rouge_metric_from_single_example(reference_summary, baseline_summary)
            custom_scores = diag.rouge_metric_from_single_example(reference_summary, custom_summary)

            # accumulate
            for metric in metrics_to_track:
                for stat in stats:
                    # pattern: get the current accumulated value for each stat within each metric
                    #       then add the new scores from the current example and continue to the next example
                    accumulated_baseline[metric][stat] += getattr(baseline_scores[metric], stat)
                    accumulated_custom[metric][stat] += getattr(custom_scores[metric], stat)

    # Calculate averages by dividing each stat within each metric
    avg_baseline = {metric: {stat: accumulated_baseline[metric][stat] / num_examples for stat in stats} for metric in metrics_to_track}
    avg_custom = {metric: {stat: accumulated_custom[metric][stat] / num_examples for stat in stats} for metric in metrics_to_track}
    
    return avg_baseline, avg_custom

avg_baseline, avg_custom = evaluate_models(test_split, num_examples=100)

metric_dict = MetricDictionary()

# mock dict template from plot.py
mapping = {
    "rouge1": "rouge-1",
    "rouge2": "rouge-2",
    "rougeL": "rouge-len"
}
stat_mapping = {
    "precision": "precision",
    "recall": "recall",
    "fmeasure": "f1"
}

#use above mock to populate the actual metric dictionary
for m_orig, m_new in mapping.items():
    for s_orig, s_new in stat_mapping.items():
        """
        Similar to the double fors and dict comprehensions used above, just loop trough
        to access each individual stat, and then add the corresponding accumulated average score
        from above to that specific stat in the metric dictionary.
        """
        metric_dict.add_metric("baseline", m_new, s_new, avg_baseline[m_orig][s_orig])
        metric_dict.add_metric("custom", m_new, s_new, avg_custom[m_orig][s_orig])

print("\nEvaluation complete. Average Scores:")
print(f"Baseline: {avg_baseline}")
print(f"Custom: {avg_custom}")

print("\nGenerating plots in 'generated_plots/'...")
# create the plot
create_bar_graph(metric_dict.get())




