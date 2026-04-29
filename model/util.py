from typing import Sequence

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from model import NO_WORD
from model import NO_LINK_TYPE
from model.lg_parser.lg_parser import parse_sentence_features, split_sentence_spans
from model.model_diagnostics.model_diagnostics import DiagnosticCapture
from model.token_link_translation.token_link_translation import (
    build_token_to_word_mapping,
    build_word_pair_matrices,
    expand_word_pair_matrices_to_tokens,
)

# NOTE: lg parses can fail to generate certain individual links


"""
Notes on batching:

Normally a batch would imply multiple full text examples being processed together.

But here, since the link grammar parsing by default operates on a sentence at a time,
we treat each sentence as its own batch.

1 Batch
---------------------
Originally one full text prompt -> is now one sentence "span"

This may cause some difficulties with organizing a training regement since it's typical
to train on multiple full examples each representing a batch. So it may be necessary to
reorganize the training loop to accommodate this, or to find a way to batch multiple sentences together.
But we can take a look at that when it comes up.
"""

"""
-----------------------------------------------------------------------
ATTENTION MECHANISM
-----------------------------------------------------------------------

Custom Attention Mechanism adhering to Hugging Face's AttentionInterface.

This replaces the standard multi-head self-attention with one that biases
attention weights using the token distance and link type matrices that come
from the link grammar parse.
"""
def linkgram_attention(
    #reference to the current module
    module,
    #query is passed automatically on use
    query,
    #key is passed automatically on use
    key,
    #value is passed automatically on use
    value,
    #optional: may want to use if we want to train via attention masking
    attention_mask=None,
    #largely irrelevant in our use case, just passing it in to adhere to the expected interface
    dropout=0.0,
    #this is passed by the model, and represents the sqrt(d_k) factor from the attention formula
    scaling=1.0,
    is_causal=False,
    #IMPORTANT: used to pass in our custom link grammar features, and
    #       also contains model relevant data which is passed from module to module
    #       during inference and training
    **kwargs,
):

    #Instantiate the diagnostic capture object to log intermediate values for analysis
    diagnostic_capture = DiagnosticCapture()

    # compute the baseline attention scores
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    
    # if this is an encoder layer and it has our biases attached, apply the link grammar bias
    if getattr(module, "is_decoder", False) == False and hasattr(module, "distance_bias"):
        distance = getattr(module, "token_distance_matrix", None)
        link_type = getattr(module, "token_link_type_matrix", None)
        
        if distance is not None and link_type is not None:
            #ensure matrices are longs
            distance = distance.long()
            link_type = link_type.long()

            # use the token distances to gather the learned distance bias for each token pair.
            valid_distance_mask = distance.ge(0) # ge -> greater than or equal to 
            distance_ids = distance.clamp(0, module.distance_bias.num_embeddings - 1) # clamp -> limits the values in distance to be between 0 and num_embeddings - 1,
                                                                                      #        so that they can be used as indices for the embedding lookup.
            #This gets the "attached" distance bias
            dist_bias = module.distance_bias(distance_ids) * valid_distance_mask.unsqueeze(-1) # unsqueeze is used here for tensor shaping, 
            diagnostic_capture.from_tensor(dist_bias, "Link Type Bias")
                                                                                               #       module.distance_bias is from the injected embedding layer
            # for directly linked words, add the learned link type bias as well.
            #       Get the "attached" link type bias
            direct_link_mask = link_type.ne(NO_LINK_TYPE) # ne -> not equal to
            valid_link_type_mask = link_type.clamp(0, module.link_type_bias.num_embeddings - 1)
            #uses link type bias to ensure that only valid link types contribute to the bias
            link_bias = module.link_type_bias(valid_link_type_mask) * direct_link_mask.unsqueeze(-1)
            diagnostic_capture.from_tensor(link_bias, "Link Type Bias")
            
            # sum the biases and permute to match (batch, heads, seq, seq)
            total_bias = (dist_bias + link_bias).permute(0, 3, 1, 2)
            diagnostic_capture.from_tensor(total_bias, "Total Link Bias")
            attention_scores = attention_scores + total_bias

    # Apply the attention mask if there is one
    #       In our case there likely won' tbe
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    
    # softmax and define dropout
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = torch.nn.functional.dropout(attention_weights, p=dropout, training=module.training)

    # apply the attention weights to the values and project back to the model dimension.
    #       final step as layed out in project proposal
    attention_output = torch.matmul(attention_probs, value)
    attention_output = attention_output.transpose(1, 2).contiguous()
    
    return attention_output, None

# register our custom attention globally so it can be "attached" to the model later by name
AttentionInterface.register("linkgram", linkgram_attention)


ALL_MASK_ATTENTION_FUNCTIONS.register("linkgram", ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["eager"])
"""
-----------------------------------------------------------------------
TENSOR MAPPING INTERFACE
-----------------------------------------------------------------------
"""

"""
Builds the token level matrices for one example in the batch.

* truncates the raw text to match what the tokenizer kept
* parses each sentence sized span with link grammar
* gathers the word level links and word spans
* expands those word level relationships to token level matrices
"""
def build_single_example_linkgram_matrices(
    text: str,
    offset_mapping: torch.Tensor,
    max_distance: int,
    device: torch.device,
    link_type_to_id: dict[tuple[str, str], int],
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = offset_mapping.shape[0]
    neutral_distance_matrix = torch.full((seq_len, seq_len), -1, dtype=torch.long, device=device)
    neutral_link_type_matrix = torch.full(
        (seq_len, seq_len),
        NO_LINK_TYPE,
        dtype=torch.long,
        device=device,
    )

    # the tokenizer may truncate the text, so only parse the region that still has tokens.
    # this avoids computing features/matrices for text that is ignored by the model from the start.
    max_char_end = int(offset_mapping[:, 1].max().item()) if seq_len > 0 else 0
    truncated_text = text[:max_char_end]
    if not truncated_text.strip():
        return neutral_distance_matrix, neutral_link_type_matrix

    word_spans: list[tuple[int, int]] = []
    links: list[tuple[int, int, int]] = []

    # parse each sentence sized span separately so that we do not create links across sentence boundaries.
    #       uses the lg parser utils defined at bottom of file
    for sentence_start, sentence_end in split_sentence_spans(truncated_text):
        sentence_word_spans, sentence_links = parse_sentence_features(
            truncated_text[sentence_start:sentence_end],
            sentence_start,
            link_type_to_id,
        )

        # each sentence uses local word indices, so shift them into the full example index space.
        word_offset = len(word_spans)
        word_spans.extend(sentence_word_spans)
        links.extend(
            (left_word + word_offset, right_word + word_offset, link_type_id)
            for left_word, right_word, link_type_id in sentence_links
        )

    token_to_word = build_token_to_word_mapping(offset_mapping, word_spans, device)
    if not word_spans:
        return neutral_distance_matrix, neutral_link_type_matrix

    # after the word level matrices are built, expand them back out to token level.
    word_distance_matrix, word_link_type_matrix = build_word_pair_matrices(
        num_words=len(word_spans),
        links=links,
        unreachable_distance=max_distance,
        device=device,
    )
    return expand_word_pair_matrices_to_tokens(
        token_to_word,
        word_distance_matrix,
        word_link_type_matrix,
    )

"""
-----------------------------------------------------------------------
INFERENCE TIME INTERFACE
-----------------------------------------------------------------------

These are methods called when we perform inference or training
"""

"""
This is a utility function to be run before the model forward definition.

This not only tokenizes the input text, but also computes the two token level
matrices that our custom attention mechanism expects.
"""
def prepare_linkgram_inputs(
    texts: str | Sequence[str],
    tokenizer,
    *,
    max_length: int,
    max_distance: int,
    #optional
    device: torch.device | str | None = None,
    #optional
    link_type_to_id: dict[tuple[str, str], int] | None = None,
):
    # if a single string is passed in, wrap it so that the rest of the function can treat it like a batch.
    batch_texts = [texts] if isinstance(texts, str) else list(texts)
    tokenized = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    # build the matrices on the same device that they will later be used on.
    matrix_device = torch.device(device) if device is not None else tokenized["input_ids"].device
    link_type_to_id = {} if link_type_to_id is None else link_type_to_id
    token_distance_matrices = []
    token_link_type_matrices = []

    # compute the token level features for each example separately and then stack them back into a batch.
    for batch_index, text in enumerate(batch_texts):
        token_distance_matrix, token_link_type_matrix = build_single_example_linkgram_matrices(
            text=text,
            offset_mapping=tokenized["offset_mapping"][batch_index],
            max_distance=max_distance,
            device=matrix_device,
            link_type_to_id=link_type_to_id,
        )
        token_distance_matrices.append(token_distance_matrix)
        token_link_type_matrices.append(token_link_type_matrix)

    # offset_mapping is only needed for feature construction, so remove it before returning.
    token_distance_matrix = torch.stack(token_distance_matrices, dim=0)
    token_link_type_matrix = torch.stack(token_link_type_matrices, dim=0)
    tokenized.pop("offset_mapping")

    if device is not None:
        tokenized = tokenized.to(device)

    return tokenized, token_distance_matrix, token_link_type_matrix, link_type_to_id

"""
Utility function to inject our custom linkgram biases into the encoder's standard BartAttention modules.
"""
def inject_linkgram_attention(model, num_link_types: int, max_distance: int):
    config = model.config
    encoder = model.model.encoder
    num_heads = config.encoder_attention_heads
    
    # tell the model to use our custom attention function
    config._attn_implementation = "linkgram"
    
    # attach the bias embeddings directly to the encoder's self-attention modules
    for layer in encoder.layers:
        attn = layer.self_attn
        
        attn.distance_bias = nn.Embedding(max_distance + 1, num_heads)
        attn.link_type_bias = nn.Embedding(num_link_types, num_heads)
        
        # initialize with zeros so it acts exactly like baseline BART before training
        nn.init.zeros_(attn.distance_bias.weight)
        nn.init.zeros_(attn.link_type_bias.weight)
        
        # Move the embeddings to the same device and dtype as the attention projection weights
        attn.distance_bias.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)
        attn.link_type_bias.to(device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype)

    print("Successfully injected LinkGram biases into BART Encoder.")

"""
Attaches the token distance and link type matrices to the encoder's self attention modules
so they can be read by our custom attention function during the forward pass.
"""
def attach_linkgram_matrices(model, token_distance_matrix: torch.Tensor, token_link_type_matrix: torch.Tensor):
    for layer in model.model.encoder.layers:
        layer.self_attn.token_distance_matrix = token_distance_matrix
        layer.self_attn.token_link_type_matrix = token_link_type_matrix

