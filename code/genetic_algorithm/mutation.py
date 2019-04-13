import primitive
import create_population
from copy import deepcopy

def mutate_rand_tree(x):
    if x.get_tokens() <= 2:
        return x
    x_ = deepcopy(x)
    n1, id1 = x_.get_random()
    n1.nodes[id1] = create_population.create_random_model(3)
    return x_