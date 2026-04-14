"""
Custom Attention Mechanism for to place within a BART Model.

Here we define the piece of model architecture that we will use to replace the standard attention mechanism found in baseline_model/model.py.
Essentially, we will be hot-swapping the original attention mechanism with this one.
"""

import torch
from torch import nn

# CONSTANTS
NO_WORD = -1
NO_LINK_TYPE = -1

class LinkGramAttention(nn.Module):

    """
    Custom Attention Mechanism for to place within a BART Model.

    This replaces the standard multi-head self-attention with one that biases
    attention weights using the token distance and link type matrices that come
    from the link grammar parse.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_distance: int,
        num_link_types: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # dim(heads) must be divisible by num_heads for multi-head attention to work
        #       this allows for the parallelization
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.max_distance = max_distance  # maximum distance for link bias
        self.num_link_types = num_link_types
        self.scaling = self.head_dim ** -0.5  # scaling factor for attention scores

        #Q, K, V values that will be instantiated with the state of the old attention layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # initialize our trainable biases for distance and link type with zeros
        self.distance_bias = nn.Embedding(max_distance + 1, num_heads)
        self.link_type_bias = nn.Embedding(num_link_types, num_heads)
        nn.init.zeros_(self.distance_bias.weight)
        nn.init.zeros_(self.link_type_bias.weight)


    """
    Utility methods for the attention mechanism.
    """
    # copies attention module weights from arg: attention_module
    def copy_projection_weights_from(self, attention_module: nn.Module) -> None:
        self.q_proj.load_state_dict(attention_module.q_proj.state_dict())
        self.k_proj.load_state_dict(attention_module.k_proj.state_dict())
        self.v_proj.load_state_dict(attention_module.v_proj.state_dict())
        self.out_proj.load_state_dict(attention_module.out_proj.state_dict())

    def build_attention_bias(
        self,
        token_distance_matrix: torch.Tensor,
        token_link_type_matrix: torch.Tensor,
        no_link_type: int = NO_LINK_TYPE,
    ) -> torch.Tensor:
        # the two matrices describe the same token pairs, so they must be the same shape.
        if token_distance_matrix.shape != token_link_type_matrix.shape:
            raise ValueError("token_distance_matrix and token_link_type_matrix must have the same shape")

        # allow either a single example matrix or a full batch of matrices.
        if token_distance_matrix.ndim == 2:
            token_distance_matrix = token_distance_matrix.unsqueeze(0)
            token_link_type_matrix = token_link_type_matrix.unsqueeze(0)
        elif token_distance_matrix.ndim != 3:
            raise ValueError("token pair matrices must have shape (seq_len, seq_len) or (batch_size, seq_len, seq_len)")

        # use the token distances to gather the learned distance bias for each token pair.
        valid_distance_mask = token_distance_matrix.ge(0)
        distance_ids = token_distance_matrix.clamp(0, self.max_distance)
        bias = self.distance_bias(distance_ids) * valid_distance_mask.unsqueeze(-1)

        # for directly linked words, add the learned link type bias as well.
        direct_link_mask = token_link_type_matrix.ne(no_link_type)
        if torch.any(direct_link_mask):
            invalid_link_types = token_link_type_matrix[direct_link_mask]
            if torch.any(invalid_link_types < 0) or torch.any(invalid_link_types >= self.num_link_types):
                raise ValueError("token_link_type_matrix contains an out-of-range link type id")

            safe_link_ids = token_link_type_matrix.clamp(0, self.num_link_types - 1)
            bias = bias + self.link_type_bias(safe_link_ids) * direct_link_mask.unsqueeze(-1)

        return bias.permute(0, 3, 1, 2).contiguous()
    
    # see the shape of the output of the projection layeres
    def _shape(self, tensor: torch.Tensor, batch_size: int, seq_len: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    """
    Public interface for the attention mechanism.

    Inputs and Outputs for the layer.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_distance_matrix: torch.Tensor | None = None,
        token_link_type_matrix: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        no_link_type: int = NO_LINK_TYPE,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        """
        enforces and defines the expected shapes of input/output tensors.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape (batch_size, seq_len, embed_dim)")
        batch_size, seq_len, embed_dim = hidden_states.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"expected hidden size {self.embed_dim}, got {embed_dim}")

        """
        If no token level features are passed in, build neutral matrices so the
        layer still behaves like normal attention.
        
        If the matrices are passed in, move them onto the same device as the
        hidden states before using them.
        """
        if token_distance_matrix is None and token_link_type_matrix is None:
            token_distance_matrix = torch.full(
                (batch_size, seq_len, seq_len),
                -1,
                dtype=torch.long,
                device=hidden_states.device,
            )
            token_link_type_matrix = torch.full(
                (batch_size, seq_len, seq_len),
                no_link_type,
                dtype=torch.long,
                device=hidden_states.device,
            )
        elif token_distance_matrix is None or token_link_type_matrix is None:
            raise ValueError("token_distance_matrix and token_link_type_matrix must be provided together")
        else:
            token_distance_matrix = token_distance_matrix.to(hidden_states.device)
            token_link_type_matrix = token_link_type_matrix.to(hidden_states.device)

        # project the hidden states into Q, K, and V and reshape for multi-head attention.
        query_states = self._shape(self.q_proj(hidden_states), batch_size, seq_len)
        key_states = self._shape(self.k_proj(hidden_states), batch_size, seq_len)
        value_states = self._shape(self.v_proj(hidden_states), batch_size, seq_len)

        # compute the baseline attention scores and then add the link grammar bias.
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        attention_scores = attention_scores + self.build_attention_bias(
            token_distance_matrix,
            token_link_type_matrix,
            no_link_type=no_link_type,
        )
        
        # Apply the attention mask if there is one
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = torch.nn.functional.dropout(attention_weights, p=self.dropout, training=self.training)

        # apply the attention weights to the values and project back to the model dimension.
        attention_output = torch.matmul(attention_probs, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attention_output = self.out_proj(attention_output)

        if output_attentions:
            return attention_output, attention_weights

        return attention_output, None
