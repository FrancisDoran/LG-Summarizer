import torch

from model.model_diagnostics.model_diagnostics import DiagnosticCapture
from model.util import prepare_linkgram_inputs

"""
DataCollator

This class allows "streaming" of the dataset and linkgram information
and passing it to the model in batches as each example is needed.

This avoids the need for precomputing which would require 
computing and storing token embeddings, and link information
for the entire dataset (or at least for more than we would use).
"""

class LinkGramDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int,
        max_distance: int,
        link_type_to_id: dict,
        device: torch.device | str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_distance = max_distance
        self.link_type_to_id = link_type_to_id
        self.device = device
    
    # allows for treating object of type LinkGramDataCollator as a function
    def __call__(self, examples):
        # Extract articles and highlights
        articles = [ex["article"] for ex in examples]
        highlights = [ex["highlights"] for ex in examples]

        tokenized, dist_matrix, link_matrix, _ = prepare_linkgram_inputs(
            articles,
            self.tokenizer,
            max_length=self.max_length,
            max_distance=self.max_distance,
            device=self.device,
            link_type_to_id=self.link_type_to_id,
        )

        diagnostic_capture = DiagnosticCapture()
        diagnostic_capture.get_tensor_norm(link_matrix, diagnostic_name="Link Type Matrix Norm")

        #tokenize highlights for labels
        labels = self.tokenizer(
            text_target=highlights,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        tokenized["labels"] = labels["input_ids"]
        tokenized["token_distance_matrix"] = dist_matrix
        tokenized["token_link_type_matrix"] = link_matrix

        return tokenized
