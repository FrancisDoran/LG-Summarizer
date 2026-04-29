from collections import deque
from typing import Sequence

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from model.lg_parser.lg_parser import parse_sentence_features, split_sentence_spans
from model.model_diagnostics.model_diagnostics import DiagnosticCapture

#constants
NO_WORD = -1
NO_LINK_TYPE = -1

# lg parses can fail to generate certain individual links

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
def _build_single_example_linkgram_matrices(
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

    token_to_word = _build_token_to_word_mapping(offset_mapping, word_spans, device)
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
TENSOR MAPPING UTILS
-----------------------------------------------------------------------
called from the tensor mapping interface

"""

"""
Breadth First Search implementation

Used to compute the distance between each pair of words in the lg parse.
We incorporate distance into the bias we add to the attention scores.

args:
    adjacency:
        [node_index, [neighbor_indices]]
    source:
        starting node
    unreachable_distance:
        fallback value
"""
def _bfs_distances(adjacency: list[list[int]], source: int, unreachable_distance: int) -> list[int]:
    # allocate a list the same size as adjacency arg, and fill it with -1 (placeholder)
    distances = [-1] * len(adjacency)
    distances[source] = 0
    #deque with first node appended, deque allows for efficient pops from the left
    queue = deque([source])

    while queue:
        #get first element from left
        node = queue.popleft()
        #use passed in adjacency to loop through the neighbors of current node
        for neighbor in adjacency[node]:
            #skip if already visited
            if distances[neighbor] != -1:
                continue
            
            #otherwise assign it the distance of the the current source node + 1 (1 step further in the graph)
            distances[neighbor] = distances[node] + 1
            #add current children to the deque to be traversed
            queue.append(neighbor)
    
    # for any nodes not traversed, assign the fallback unreachable_distance value passed in, otherwise return the distance calculated above
    return [unreachable_distance if distance == -1 else distance for distance in distances]

"""
Maps tokenizer offsets back to the word indices gathered from the link grammar parse.
    * does this by checking how much overlap there is between the characters of the word and the characters of the token
    * If a token does not overlap any parsed word, it will remain as NO_WORD.
-------------------------------------------------------------------------------------------------------
ex...

original text (sentence in our case): "This food tastes unbelievable."
tokenization: ["This", "food", "tastes", "un", "believable", "."]
    * note: the way we parse w/ lg rn removes punctuation, something to be worked on
    * although we don't factor in punctuation for link grammar, it still gets factored in by the model itself

resulting map: [0, 1, 2, 3, 3, NO_WORD]
-------------------------------------------------------------------------------------------------------
this mapping is used later as a mask to expand the word level link grammar features
into their respective token level tensor formats

would ordered traversal be better?
"""
def _build_token_to_word_mapping(
    #offset_mapping contains character start and end positions of each token in the original text, index is the token number, value is (char_start, char_end)
    #    * in our case we parse each sentence separately, so the offsets are relative to the start of the sentence span
    offset_mapping: torch.Tensor,
    word_spans: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    #allocate a tensor
    token_to_word = torch.full((offset_mapping.shape[0],), NO_WORD, dtype=torch.long, device=device)
    #if no word spans parsed, return the placeholder tensor (attention bias will have no effect)
    if not word_spans:
        return token_to_word
    
    #loop primer, keeps track of current word span being looked at
    candidate_start = 0
    # iterate collecting elements in the form returned by the tokenizer when we set return_offsets_mapping=True,
    #       which gives us the character start and end positions of each token in the original text.
    for token_index, (token_start, token_end) in enumerate(offset_mapping.tolist()):
        # skip special tokens and padding which are represented by empty spans.
        if token_start >= token_end:
            continue

        """
        This loop keeps the current word being looked at, and the current token being looked at, aligned.

        The original for loop itself does not increment candidate_start (the current word) in order to allow for multiple tokens in a row
        to be properly matched with the correct word (subword token cases)

        So we have a value representing the current word, and increment it only in the event that the current token starts after the end of the current word, which means we need to move on to the next word.
        """
        while candidate_start < len(word_spans) and word_spans[candidate_start][1] <= token_start:
            candidate_start += 1
        
        #current best word index
        best_word_index = NO_WORD
        #current best overlap
        best_overlap = 0
        probe_index = max(0, candidate_start - 1)
        
        # iterate over the words
        while probe_index < len(word_spans):
            # for each word, get the first and last character positions of the word in the original sentence
            word_start, word_end = word_spans[probe_index]
            # again, this is for subword token cases. While looping through the words, if we find a word that starts after the current token ends,
            #       then we have encountered the end of a sub token and should break.
            if word_start >= token_end and best_overlap > 0:
                break
            
            #compute the chracter overlap between token and word, and update the current best placeholders above
            overlap = min(token_end, word_end) - max(token_start, word_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_word_index = probe_index

            if word_start > token_end:
                break
            probe_index += 1
        
        # after looping through the words, if we have found any overlap at all, assign the token to the best word index found.
        if best_word_index != NO_WORD:
            token_to_word[token_index] = best_word_index

    # return the found mapping
    return token_to_word

"""
This function builds adjacency matrices for two relationships...
* Relationship 1: The distance between a pair of words in the link grammar parse
* Relationship 2: The type of link, or lack thereof, between a pair of words in the link grammar parse

This returns the two matrices as PyTorch tensors which can then be used in the forward pass of our LinkGramAttention module.
"""
def build_word_pair_matrices(
    num_words: int,
    # this is a list of tuples, where each tuple is (left_word_index, right_word_index, link_type_id)
    links: Sequence[tuple[int, int, int]],
    # this is the distance that will be assigned to pairs of words that are not reachable from each other in the lg parse. 
    unreachable_distance: int | None = None,
    # this is the device to use when computing. If None, the matrices will be created on the CPU.
    #       Otherwise, they will be created on the specified device.
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Early returns and error handling for invalid inputs:
        * num_words must be positive
        * unreachable_distance must be non-negative
    """
    if num_words <= 0:
        raise ValueError("num_words must be positive")
    if unreachable_distance is None:
        unreachable_distance = num_words
    if unreachable_distance < 0:
        raise ValueError("unreachable_distance must be non-negative")
    
    # Use the num words argument to allocate memory for the adjacency matrices.
    adjacency = [[] for _ in range(num_words)]

    """
    Create a MxN tensor that becomes the link type matrix.

    * size
    * value to be filled in when no link type is found
    * datatype
    * device to be allocated on
    """
    link_type_matrix = torch.full(
        (num_words, num_words),
        NO_LINK_TYPE,
        dtype=torch.long,
        device=device,
    )
    
    # traverse through the list of links gathered from the lg parse, and populate the adjacency list with it.
    for left_word, right_word, link_type_id in links:

        """
        Early returns and error handling for invalid link tuples:
        """
        if not (0 <= left_word < num_words and 0 <= right_word < num_words):
            raise ValueError(
                f"link ({left_word}, {right_word}, {link_type_id}) references an invalid word index"
            )
        if link_type_id < 0:
            raise ValueError("link_type_id must be non-negative")

        adjacency[left_word].append(right_word)
        adjacency[right_word].append(left_word)
        #Square matrix where each entry (i, j) corresponds to the link type between word i and word j, or NO_LINK_TYPE if there is no link.
        link_type_matrix[left_word, right_word] = link_type_id
        link_type_matrix[right_word, left_word] = link_type_id

    # Do a similar process as above except for the distance matrix now.
    distance_rows = [
        torch.tensor(
            _bfs_distances(adjacency, source, unreachable_distance),
            dtype=torch.long,
            device=device,
        )
        for source in range(num_words)
    ]
    distance_matrix = torch.stack(distance_rows, dim=0)

    #Shapes... 
    #       distance_matrix: (num_words, num_words)
    #       link_type_matrix: (num_words, num_words)
    return distance_matrix, link_type_matrix


"""
The two following functions are helper functions which
ensure that the inputs to our main function, expand_word_pair_matrices_to_tokens,
are in the correct shape.

If unbatched input is received, add a batch dimension of size 1
If already batched input is received, return as is.

token_to_word maps individual tokens to word indices
"""
def _ensure_batched_mapping(token_to_word: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if token_to_word.ndim == 1:
        return token_to_word.unsqueeze(0), True
    if token_to_word.ndim == 2:
        return token_to_word, False
    raise ValueError("token_to_word must have shape (seq_len,) or (batch_size, seq_len)")


def _ensure_batched_square_matrix(
    matrix: torch.Tensor,
    batch_size: int,
    name: str,
) -> torch.Tensor:
    """
    If the input matrix is 2D
            we assume it's the same for all examples in the batch and expand it.

    else
            we check that it's already batched properly and return it as is.
    """
    if matrix.ndim == 2:
        return matrix.unsqueeze(0).expand(batch_size, -1, -1)
    if matrix.ndim == 3 and matrix.shape[0] == batch_size:
        return matrix
    raise ValueError(f"{name} must have shape (n, n) or (batch_size, n, n)")

"""
Processes the previously made link type and word distance adjacency matrices,
and expands them to token-level pair features that can be used in the attention
bias computation of our LinkGramAttention module. And this is done by using the
token_to word mapping which is calculated when we tokenize the input text.

The attention logits produced here will be added as biases to the attention scores.
"""
def expand_word_pair_matrices_to_tokens(
    # Mapping from token to word index Shape = (batch_size, seq_len) or (seq_len,)
    token_to_word: torch.Tensor,
    # Adjacency matrices for word-level relationships, 
#       Shape = (batch_size, num_words, num_words) or (num_words, num_words)
    word_distance_matrix: torch.Tensor,
    word_link_type_matrix: torch.Tensor,
    # Filler values to use for positions where token_to_word is no_word,
    #       or where word_distance_matrix and word_link_type_matrix have no link.
    no_word: int = NO_WORD,
    no_link_type: int = NO_LINK_TYPE,
    padding_distance: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expand word-level parse features to token-level pair features.

    `token_to_word` maps each token position to a word index. Use `no_word` for
    special tokens and padding so those positions receive a neutral bias.
    """
    token_to_word, squeezed = _ensure_batched_mapping(token_to_word)
    batch_size, seq_len = token_to_word.shape
    
    """
    Shape the word level matrices
    """
    word_distance_matrix = _ensure_batched_square_matrix(
        word_distance_matrix,
        batch_size,
        "word_distance_matrix",
    ).to(token_to_word.device)
    word_link_type_matrix = _ensure_batched_square_matrix(
        word_link_type_matrix,
        batch_size,
        "word_link_type_matrix",
    ).to(token_to_word.device)
    
    """
    Allocate the tensors for the token level matrices.

    Initialize with padding (filler) values
    """
    token_distance_matrix = torch.full(
        (batch_size, seq_len, seq_len),
        padding_distance,
        dtype=torch.long,
        device=token_to_word.device,
    )
    token_link_type_matrix = torch.full(
        (batch_size, seq_len, seq_len),
        no_link_type,
        dtype=torch.long,
        device=token_to_word.device,
    )

    # Iterate over the batches received
    for batch_idx in range(batch_size):
        #Get the mapping for token_to_word for one example in the batch (one sentence in the full example)
        mapping = token_to_word[batch_idx]
        #use the mapping to give each valid token an index in valid_positions
        #       finds all valid tokens
        #
        #squeeze is used to remove the extra dimension added by nonzero,
        #       so that valid_positions is a 1D tensor of token indices instead
        #       of MxN where N is the number of dimensions of the mapping.
        valid_positions = torch.nonzero(mapping.ne(no_word), as_tuple=False).squeeze(-1)
        # if number of elements is = 0, continue to the next example in the batch,
        #       since there are no valid tokens to process in this example.
        if valid_positions.numel() == 0:
            continue

        """ 
        A valid example at this point
        
        toekn_to_word mapping = [0, 1, 1, 2, -1]
        valid_positions = [0, 1, 2, 3]

        token_to_word[valid_positions]  = [0, 1, 1, 2]

        """
        
        """
        TODO: refine this explanation
        word_ids        = [0, 1, 1, 2]
        """
        word_ids = mapping[valid_positions]

        """
        TODO: refine this explanation
        valid_positions[:, None] -> column shape (4, 1)
        valid_positions[None, :] -> row shape    (1, 4)
        """
        token_distance_matrix[batch_idx][valid_positions[:, None], valid_positions[None, :]] = (
            word_distance_matrix[batch_idx][word_ids[:, None], word_ids[None, :]]
        )
        # link types are stored as they are encountered in the link grammar parse
        # normalization of the link types is learned by the Embedding layer we use to represent the biases.
        token_link_type_matrix[batch_idx][valid_positions[:, None], valid_positions[None, :]] = (
            word_link_type_matrix[batch_idx][word_ids[:, None], word_ids[None, :]]
        )
    
    #Return those token_level matrices
    if squeezed:
        return token_distance_matrix[0], token_link_type_matrix[0]

    return token_distance_matrix, token_link_type_matrix

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
        token_distance_matrix, token_link_type_matrix = _build_single_example_linkgram_matrices(
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

