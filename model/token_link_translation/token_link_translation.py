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

from collections import deque
from typing import Sequence

import torch

from model import NO_WORD
from model import NO_LINK_TYPE

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
def build_token_to_word_mapping(
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
def ensure_batched_mapping(token_to_word: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if token_to_word.ndim == 1:
        return token_to_word.unsqueeze(0), True
    if token_to_word.ndim == 2:
        return token_to_word, False
    raise ValueError("token_to_word must have shape (seq_len,) or (batch_size, seq_len)")


def ensure_batched_square_matrix(
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
    token_to_word, squeezed = ensure_batched_mapping(token_to_word)
    batch_size, seq_len = token_to_word.shape
    
    """
    Shape the word level matrices
    """
    word_distance_matrix = ensure_batched_square_matrix(
        word_distance_matrix,
        batch_size,
        "word_distance_matrix",
    ).to(token_to_word.device)
    word_link_type_matrix = ensure_batched_square_matrix(
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

