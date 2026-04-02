"""
Custom Attention Mechanism for to place within a BART Model.

Here we define the piece of model architecture that we will use to replace the standard attention mechanism found in baseline_model/model.py.
Essentially, we will be hot-swapping the original attention mechanism with this one.

Because attention mechanisms individually are their own type of neural network, we will need to train our custom mechanism separately
as the original BART model will come pretrained which is missing from our custom mechanism.
* We could attempt to port the weights from the original attention mechanism to our custom one, 
  but this is not guaranteed to work and may require additional fine-tuning.
* If that fails, we can train our custom attention mechanism from scratch using a dataset of input-output pairs that are relevant to our task.

If we choose to train from scratch, it is important to mention this in the final paper. Mentioning procedure, dataset, etc..

Design...
* Our attention mechanism will be a class that inherits from nn.Module, which is standard practice for PyTorch.
* Within this class we will define define, in the constructor, the logic for our attention mechanism.
    * Here we can implement the math from scratch (not terribly difficult with python), or we can use 
      existing PyTorch methods, math modules to speed up the process (Nothing is lost as a result of using
      built-ins here as far as I know so this is up to developer discretion and will.
* Finally we will define a forward method which will take in the layer inputs, apply our defined logic to it, and then return
  the output in a format we define within this same forward method.
"""

import torch
from torch import nn

class LinkGramAttention(nn.Module):

    # IMPLEMENT MATH FOR LINKGRAM ATTENTION MECHANISM HERE
    def __init__(self, input_dim, output_dim):
        super(LinkGramAttention, self).__init__()

    # Implement forward pass here, make sure this method returns a tensor that matches the shape of the original attention mech.
    def forward(self, input):

        output = None # REPLACE WITH ACTUAL OUTPUT
        return output


