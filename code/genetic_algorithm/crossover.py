from copy import deepcopy

def crossfit(x, y):
    if x.get_tokens() <= 2 or y.get_tokens() <= 2:
        return x
    x_ = deepcopy(x)
    y_ = deepcopy(y)
    n1, id1 = x_.get_random()
    n2, id2 = y_.get_random()
    n1.nodes[id1], n2.nodes[id2] = n2.nodes[id2], n1.nodes[id1]
    return x_