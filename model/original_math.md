
File contains a pure, non-negotiable implementation of the math for our attention mechanism.

Saving here in a md in case the code strays or we need to refer to it.

```python
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
```

.shape is only available for values that are tensors/np arrays or that
       are implemented as children of those types

```python
def link_bias(tokens, graph, is_linked_bias, link_type_bias_dict):
    c = 0
    for i in tokens:
        for j in tokens:

            qi = get_query(i)

            n_i = word_to_node(token_to_word(i))
            n_j = word_to_node(token_to_word(j))

            g = shortest_distance(graph, n_i, n_j, total_vertices)
            
            # is_linked_bias is a trainable parameter
            sij = (qi @ kj)/np.sqrt(kj.shape[0]) + is_linked_bias/(g+1)

            alpha = torch.nn.Softmax(sij)

            if g == 1:
                r = link_type_bias_dict[token_to_link(i,j)]
            else:
                r = np.zeros(vj.shape())

            vrj = vj + r

            c += alpha*vrj
    return c
```
This should be moved to the functions that require it, and instantiated as a local variable there.
total\_vertices = len(list(linkage.words()))

