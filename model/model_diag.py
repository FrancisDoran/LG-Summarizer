"""
Module that exposes an api for calculating model diagnostics.
"""

import torch
from torch import nn

class DiagnosticCapture:
    
    #observer pattern
    _diagnostics_to_capture: list[str]

    def __init__(self):
        #observer pattern
        self._diagnostics_to_capture = list()

    # observer pattern 
    def add(self, diagnostic: str):
        self._diagnostics_to_capture.append(diagnostic)
        return self
    
    """
    Hook into a callback to execute the function and execute diagnostics from it.
    """
    def from_callback(self, callback):
        for diagnostic in self._diagnostics_to_capture:
            callback(diagnostic)

    """
    Turn a tensor into a useful diagnostic.

    Print results
    """
    def from_tensor(self, tensor: torch.Tensor, diagnostic_name: str) -> None:
        # Example diagnostic: mean and std of the tensor
        mean = tensor.mean()
        std = tensor.std()
        print(f"Diagnostic - {diagnostic_name}: mean={mean}, std={std}")
