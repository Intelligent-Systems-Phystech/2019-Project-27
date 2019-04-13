import numpy as np
import primitive
from copy import deepcopy

def create_random_model(depth):
    depth -= np.random.randint(0, 2) # construct not uniform binary trees

    if depth <= 0:
        return deepcopy(primitive.PRIMITIVES[np.random.randint(0, 2)]) # select randomly TF or IDF
    else:
        cur_primitive = deepcopy(primitive.PRIMITIVES[np.random.randint(0, len(primitive.PRIMITIVES))])
        nodes = [create_random_model(depth - 1) for _ in range(cur_primitive.valency)]
        cur_primitive.add_nodes(nodes)
        return cur_primitive


def create_population(size, max_depth):
    population = []
    for _ in range(size):
        model = create_random_model(max_depth)
        try:
            model.calc_domains()
            population.append(model)
        except primitive.DomainException as e:
            # TODO: print blank line ???
            #print(e)
            pass
        
    return population
