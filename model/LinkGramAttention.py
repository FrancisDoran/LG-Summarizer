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

from collections import deque

import linkgrammar as lg
from linkgrammar import Link, Linkage
import torch
from torch import nn

# code adapted from GeeksForGeeks
def bfs(graph, source, par, dist):
    # dequeue to store the nodes in the order they are visited
    _temp_deque = deque()
    # Mark the distance of the source node as 0
    dist[source] = 0
    # Push the source node to the queue
    _temp_deque.append(source)

    # Iterate until the queue is not empty
    while _temp_deque:
        # Pop the node at the front of the queue
        node = _temp_deque.popleft()

        # Explore all the neighbors of the current node
        for neighbor in graph[node]:
            # Check if the neighboring node is not visited
            if dist[neighbor] == float('inf'):
                # Mark the current node as the parent of the neighboring node
                par[neighbor] = node
                # Mark the distance of the neighboring node as the distance of the current node + 1
                dist[neighbor] = dist[node] + 1
                # Insert the neighboring node to the queue
                _temp_deque.append(neighbor)

def shortest_distance(graph, source, D, V):
    # par[] array stores the parent of nodes
    parent = [-1] * V

    # dist[] array stores the distance of nodes from S
    dist = [float('inf')] * V

    # Function call to find the distance of all nodes and their parent nodes
    bfs(graph, source, parent, dist)

    return dist[D]

def linkage_to_word_graph(linkage : Linkage, V : int):

    word_to_node = {}
    node_to_word = {}
    i = 0

    for word in linkage.words():
        word_to_node[word] = i
        node_to_word[i] = word
        i += 1


    edges = [ [word_to_node[link.left_word], word_to_node[link.right_word]] for link in linkage.links()]

    # Source and Destination vertex
    S, D = 2, 0

    # List to store the graph as an adjacency list
    graph = [[] for _ in range(V)]

    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    return graph

class LinkGramAttention(nn.Module):

    # IMPLEMENT MATH FOR LINKGRAM ATTENTION MECHANISM HERE
    def __init__(self, input_dim, output_dim):
        super(LinkGramAttention, self).__init__()

    # Implement forward pass here, make sure this method returns a tensor that matches the shape of the original attention mech.
    def forward(self, input):

        output = None # REPLACE WITH ACTUAL OUTPUT
        return output


def get_link_type(i : Link, j : Link):
    # bad way to do this, we should be able to establish the correct link based on the relative position of the words
    if i.right_label == j.left_label:
        return i.right_label
    if i.left_label == j.right_label:
        return i.left_label

def token_to_word():
    pass

def word_to_node():
    pass

def token_to_link():
    pass

def get_query(token):
    pass

def get_key(token):
    pass

def get_value(token):
    pass

def attention(tokens, graph, is_linked_bias, link_type_bias_dict):
    c = 0
    for i in tokens:
        for j in tokens:

            qi = get_query(i)
            kj = get_key(j)
            vj = get_value(j)

            n_i = word_to_node(token_to_word(i))
            n_j = word_to_node(token_to_word(j))

            g = shortest_distance(graph, n_i, n_j)

            sij = (qi @ kj)/np.sqrt(kj.shape[0]) + trainable_is_linked_bias/(g+1)

            m = torch.nn.Softmax()
            alpha = m(sij)

            if g == 1:
                r = link_type_bias_dict[token_to_link(i,j)]
            else:
                r = np.Zeros(vj.shape)

            vrj = vj + r

            c += alpha*vrj
    return c

V = len(list(linkage.words()))
