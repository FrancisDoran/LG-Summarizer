import numpy as np
from rouge_score import rouge_scorer
import torch

"""
Module that exposes an api for calculating model diagnostics.

Diagnostics to capture:
* Mean and standard deviation of bias tensors
* ROUGE metrics between generated summaries and reference summaries
* current step

"""

class DiagnosticCapture:

    """
    #observer pattern
    _diagnostics_to_capture: list[str]

    def __init__(self):
        #observer pattern
        self._diagnostics_to_capture = list()

    # observer pattern 
    def add(self, diagnostic: str):
        self._diagnostics_to_capture.append(diagnostic)
        return self
    
    Hook into a callback to execute the function and execute diagnostics from it.
    def from_callback(self, callback):
        for diagnostic in self._diagnostics_to_capture:
            callback(diagnostic)

    def from_tensor(self, tensor: torch.Tensor, diagnostic_name: str):
        self.bias_sum_from_tensor(tensor, diagnostic_name)
    """

    #computes and prints the mean of the input tensor
    def from_tensor(self, tensor: torch.Tensor, diagnostic_name: str):
        mean = tensor.mean()
        
        """
        print("\n" + "="*80)
        print(f"Mean of {diagnostic_name}: {mean}")
        print("="*80)
        """

    def get_tensor_norm(self, tensor: torch.Tensor, diagnostic_name: str = "Norm of Link Type Tensor"):
        norm = np.linalg.norm(tensor.cpu().numpy())
        
        """
        print("\n" + "="*80)
        print(f"{diagnostic_name}: {norm}")
        print("="*80)
        """

    
    # Generate ROUGE metrics between a generated summary and a reference summary
    def rouge_metric_from_single_example(self, reference_summary: str, generated_summary: str):

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)

        #print(f"Diagnostic - {diagnostic_name}: {scores}")
        return scores
    
    def rouge_two_model_comparison(self, reference_summary, baseline_model_generated_summary, custom_model_generated_summary):
        baseline_scores = self.rouge_metric_from_single_example(reference_summary, baseline_model_generated_summary)
        custom_scores = self.rouge_metric_from_single_example(reference_summary, custom_model_generated_summary)

        """
        print("\n" + "="*80)
        print("ROUGE Scores Comparison:")
        print(f"Baseline Model Scores: {baseline_scores}")
        print(f"Custom Model Scores: {custom_scores}")
        print("="*80)
        """
        return (baseline_scores, custom_scores)

