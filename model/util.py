from collections import deque
from functools import lru_cache
import re
from types import MethodType
from typing import Sequence

import linkgrammar as lg
from linkgrammar import Link, Sentence
import torch
from torch import nn
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutput

try:
    from .LinkGramAttention import LinkGramAttention
except ImportError:
    from LinkGramAttention import LinkGramAttention

#constants
NO_WORD = -1
NO_LINK_TYPE = -1
# this regex splits larger inputs into sentence sized spans before they are passed into link grammar.
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=(?:["\'])?[A-Z0-9])|\n+')

"""
Maps tokenizer offsets back to the word indices gathered from the link grammar parse.

If a token does not overlap any parsed word, it will remain as NO_WORD.
"""
def _build_token_to_word_mapping(
    offset_mapping: torch.Tensor,
    word_spans: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    token_to_word = torch.full((offset_mapping.shape[0],), NO_WORD, dtype=torch.long, device=device)
    if not word_spans:
        return token_to_word

    candidate_start = 0
    for token_index, (token_start, token_end) in enumerate(offset_mapping.tolist()):
        # skip special tokens and padding which are represented by empty spans.
        if token_start >= token_end:
            continue

        # move forward until we are near the first word span that could overlap this token.
        while candidate_start < len(word_spans) and word_spans[candidate_start][1] <= token_start:
            candidate_start += 1

        best_word_index = NO_WORD
        best_overlap = 0
        probe_index = max(0, candidate_start - 1)

        # look at the nearby word spans and choose the one with the greatest overlap.
        while probe_index < len(word_spans):
            word_start, word_end = word_spans[probe_index]
            if word_start >= token_end and best_overlap > 0:
                break

            overlap = min(token_end, word_end) - max(token_start, word_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_word_index = probe_index

            if word_start > token_end:
                break
            probe_index += 1

        if best_word_index != NO_WORD:
            token_to_word[token_index] = best_word_index

    return token_to_word


"""
Builds the token level matrices for one example in the batch.

This function:
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
    max_char_end = int(offset_mapping[:, 1].max().item()) if seq_len > 0 else 0
    truncated_text = text[:max_char_end]
    if not truncated_text.strip():
        return neutral_distance_matrix, neutral_link_type_matrix

    word_spans: list[tuple[int, int]] = []
    links: list[tuple[int, int, int]] = []

    # parse each sentence sized span separately so that we do not create links across sentence boundaries.
    for sentence_start, sentence_end in _split_sentence_spans(truncated_text):
        sentence_word_spans, sentence_links = _parse_sentence_features(
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
This is a utility function to be run before the model forward pass.

This not only tokenizes the input text, but also computes the two token level
matrices that our custom attention mechanism expects.
"""
def prepare_linkgram_inputs(
    texts: str | Sequence[str],
    tokenizer,
    *,
    max_length: int,
    max_distance: int,
    device: torch.device | str | None = None,
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
This section patches the forward methods of the encoder and its layers.

This is needed so that token_distance_matrix and token_link_type_matrix can be
threaded through the normal BART forward path and into our custom attention layer.
"""
def _patch_encoder_layer_forward(layer) -> None:
    if getattr(layer, "_linkgram_forward_patched", False):
        return

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        output_attentions: bool | None = False,
        token_distance_matrix: torch.Tensor | None = None,
        token_link_type_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        # this follows the original BartEncoderLayer forward logic, but now passes our extra matrices into self_attn.
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            token_distance_matrix=token_distance_matrix,
            token_link_type_matrix=token_link_type_matrix,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and not torch.isfinite(hidden_states).all():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    # replace the layer forward method in place so the rest of the model can stay unchanged.
    layer.forward = MethodType(forward, layer)
    layer._linkgram_forward_patched = True

"""
This patch does the same thing as the original BartEncoder forward method,
but also accepts and forwards our token level matrices.
"""
def _patch_encoder_forward(encoder) -> None:
    if getattr(encoder, "_linkgram_forward_patched", False):
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        token_distance_matrix: torch.Tensor | None = None,
        token_link_type_matrix: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutput:
        # keep the same output flag behavior used by the original encoder implementation.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # if embeddings were not passed in directly, gather them from the token ids now.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # create the same bidirectional encoder mask as baseline BART.
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # run each encoder layer as usual, but now include the two additional matrices.
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    token_distance_matrix=token_distance_matrix,
            token_link_type_matrix=token_link_type_matrix,
        )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )

    # replace the encoder forward method in place so generate() can keep using the same model object.
    encoder.forward = MethodType(forward, encoder)
    encoder._linkgram_forward_patched = True

"""
Utility function to replace all standard BartAttention modules in the encoder 
(and optionally decoder) with our custom LinkGramAttention.
"""
def inject_linkgram_attention(model, num_link_types: int, max_distance: int):
    config = model.config
    encoder = model.model.encoder
    # patch the encoder once so that it knows how to accept the new arguments.
    _patch_encoder_forward(encoder)
    
    #replace the encoder's self-attention
    for layer in encoder.layers:
        old_attn = layer.self_attn
        
        # Instantiate new attention module
        new_attn = LinkGramAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            max_distance=max_distance,
            num_link_types=num_link_types,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        
        # Copy pre-trained weights
        new_attn.copy_projection_weights_from(old_attn)
        # move the new module onto the same device and dtype as the module it is replacing.
        new_attn.to(device=old_attn.q_proj.weight.device, dtype=old_attn.q_proj.weight.dtype)
        # Swap
        layer.self_attn = new_attn
        # patch the layer forward so these new arguments continue through the call path.
        _patch_encoder_layer_forward(layer)

    print("Successfully injected LinkGramAttention into BART Encoder.")


"""
Breadth First Search implementation
"""
def _bfs_distances(adjacency: list[list[int]], source: int, unreachable_distance: int) -> list[int]:
    distances = [-1] * len(adjacency)
    distances[source] = 0
    queue = deque([source])

    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distances[neighbor] != -1:
                continue

            distances[neighbor] = distances[node] + 1
            queue.append(neighbor)

    return [unreachable_distance if distance == -1 else distance for distance in distances]

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
        #Get the mapping for token_to_word for one example in the batch
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
        
        mapping         = [0, 1, 1, 2, -1]
        valid_positions = [0, 1, 2, 3]

        where -1 is the no_word filler value

        and each non-negative integer corresponds to a single word in the original prompt
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
        token_link_type_matrix[batch_idx][valid_positions[:, None], valid_positions[None, :]] = (
            word_link_type_matrix[batch_idx][word_ids[:, None], word_ids[None, :]]
        )
    
    #Return those token_level matrices
    if squeezed:
        return token_distance_matrix[0], token_link_type_matrix[0]

    return token_distance_matrix, token_link_type_matrix

"""
get link type
"""
def get_link_type(i : Link, j : Link):
    # bad way to do this, we should be able to establish the correct link based on the relative position of the words
    if i.right_label == j.left_label:
        return i.right_label
    if i.left_label == j.right_label:
        return i.left_label
    raise ValueError(
        "Unable to determine link type for links with labels "
        f"({i.left_label}, {i.right_label}) and ({j.left_label}, {j.right_label})."
    )

"""
This section configures link grammar parser to output the features we need
"""
#memoize loading of the link grammar dictionary and parse options
@lru_cache(maxsize=1)
def _get_linkgrammar_runtime() -> tuple[lg.Dictionary, lg.ParseOptions]:
    return lg.Dictionary("en"), lg.ParseOptions(verbosity=0, linkage_limit=1, max_parse_time=10)

"""
Removes leading and trailing whitespace from the span boundaries.
"""
def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[end - 1].isspace():
        end -= 1
    return start, end

"""
Splits a larger text into sentence sized spans before it is passed into link grammar.

This keeps the parser working on smaller chunks and also prevents links from
being built across sentence boundaries.
"""
def _split_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sentence_start = 0

    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        start, end = _trim_span(text, sentence_start, match.start())
        if start < end:
            spans.append((start, end))
        sentence_start = match.end()

    start, end = _trim_span(text, sentence_start, len(text))
    if start < end:
        spans.append((start, end))

    return spans


"""
If link grammar fails on a sentence, fall back to simple whitespace spans.
"""
def _fallback_word_spans(text: str, text_start: int) -> list[tuple[int, int]]:
    return [
        (text_start + match.start(), text_start + match.end())
        for match in re.finditer(r"\S+", text)
    ]

"""
Parses one sentence sized span and converts it into the two things we need:
* the word spans in the original text
* the links between those words
"""
def _parse_sentence_features(
    sentence_text: str,
    sentence_start: int,
    link_type_to_id: dict[tuple[str, str], int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    dictionary, parse_options = _get_linkgrammar_runtime()

    # if parsing fails, we still return word spans so later alignment code can continue.
    try:
        linkages = Sentence(sentence_text, dictionary, parse_options).parse()
        linkage = next(iter(linkages), None)
    except Exception:
        linkage = None

    if linkage is None:
        return _fallback_word_spans(sentence_text, sentence_start), []

    word_spans: list[tuple[int, int]] = []
    local_to_compact: dict[int, int] = {}

    # gather the real words only and ignore parser specific wall tokens.
    for word_index in range(linkage.num_of_words()):
        word_text = linkage.word(word_index)
        word_start = linkage.word_char_start(word_index)
        word_end = linkage.word_char_end(word_index)
        if word_text in {"LEFT-WALL", "RIGHT-WALL"} or word_start >= word_end:
            continue

        local_to_compact[word_index] = len(word_spans)
        word_spans.append((sentence_start + word_start, sentence_start + word_end))

    links: list[tuple[int, int, int]] = []

    # convert each parser link into the compact word indices we built above.
    for link in linkage.links():
        left_word_index = lg.Clinkgrammar.linkage_get_link_lword(linkage._obj, link.index)
        right_word_index = lg.Clinkgrammar.linkage_get_link_rword(linkage._obj, link.index)
        if left_word_index not in local_to_compact or right_word_index not in local_to_compact:
            continue

        link_key = (link.left_label, link.right_label)
        link_type_id = link_type_to_id.setdefault(link_key, len(link_type_to_id))
        links.append(
            (
                local_to_compact[left_word_index],
                local_to_compact[right_word_index],
                link_type_id,
            )
        )

    return word_spans, links
