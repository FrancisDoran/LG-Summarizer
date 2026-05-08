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
    """

    """
    Sum the mean of the bias tensor and return it as a list
    """
    def bias_sum_from_tensor(self, tensor: torch.Tensor, diagnostic_name: str) -> list[float]:
        # Example diagnostic: mean and std of the tensor
        mean = tensor.mean()
        std = tensor.std()
        #print(f"Diagnostic - {diagnostic_name}: mean={mean}, std={std}")
        bias_sum = list()
        bias_sum.append(mean)

        return bias_sum

    def rouge_metric_from_single_example(self, reference_summary: str, generated_summary: str) -> dict[str, float]:

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)

        #print(f"Diagnostic - {diagnostic_name}: {scores}")
        return scores[0]


