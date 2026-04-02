"""
This module will be used to perform the matrix transformations discussed in our research proposal.

The approach here will be similar to how we are implementing our custom attention mechanism,
except that here we are only adding a layer and not replacing an existing one.

We will be using the same design approach as in our custom attention mechanism,
please read the notes at the top of model/LinkGramAttention.py for more details on the design approach.
"""

import torch
from torch import nn

class TokenTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TokenTransformer, self).__init__()

    def forward(self, x):
        transformed = None
        return transformed

