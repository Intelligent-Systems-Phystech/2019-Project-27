import primitive
import numpy as np

from get_quality import get_quality
from copy import deepcopy
from mutation import mutate_rand_tree
from crossover import crossfit
from create_population import create_random_model
from primitive import Primitives

def rank_population(population, doc_ranks, queries, query_characteristics):   

    qualities = []
    for model in population:
        quality = get_quality(model, doc_ranks, queries, query_characteristics)
        qualities.append(quality)

    return qualities

def test(doc_ranks, queries, query_characteristics):
    tmodel = deepcopy(Primitives.SQRT).add_nodes([
        deepcopy(Primitives.SQRT).add_nodes([
            deepcopy(Primitives.EXP).add_nodes([
                deepcopy(Primitives.TF)
            ])
        ])
    ])
    # tmodel = deepcopy(Primitives.ADD).add_nodes([
    #     deepcopy(Primitives.ADD).add_nodes([
    #         deepcopy(Primitives.TF),
    #         deepcopy(Primitives.IDF)
    #         ]),
    #     deepcopy(Primitives.DIV).add_nodes([
    #         deepcopy(Primitives.IDF),
    #         deepcopy(Primitives.TF)
    #     ])
    # ])

    # tmodel = deepcopy(Primitives.EXP).add_nodes([
    #     deepcopy(Primitives.MUL).add_nodes([
    #         deepcopy(Primitives.DIV).add_nodes([
    #             deepcopy(Primitives.TF),
    #             deepcopy(Primitives.TF)
    #         ]),
    #         deepcopy(Primitives.IDF)
    #     ])
    # ])

    # tmodel = deepcopy(Primitives.EXP).add_nodes([
    #     deepcopy(Primitives.ADD).add_nodes([
    #         deepcopy(Primitives.SUB).add_nodes([
    #             deepcopy(Primitives.IDF),
    #             deepcopy(Primitives.TF)
    #         ]),
    #         deepcopy(Primitives.TF)
    #     ])
    # ])

    # tmodel = deepcopy(Primitives.MUL).add_nodes([
    #     deepcopy(Primitives.EXP).add_nodes([
    #             deepcopy(Primitives.IDF)
    #     ]),
    #     deepcopy(Primitives.ADD).add_nodes([
    #             deepcopy(Primitives.TF),
    #             deepcopy(Primitives.TF)
    #     ])
    # ])

    # tmodel = deepcopy(Primitives.SQRT).add_nodes([
    #     deepcopy(Primitives.EXP).add_nodes([
    #         deepcopy(Primitives.LOG).add_nodes([
    #             deepcopy(Primitives.IDF)
    #         ])
    #     ])
    # ])
    
    print(str(tmodel))
    quality = rank_population([tmodel], doc_ranks, queries, query_characteristics)
    print(quality)

def learn_population(population, doc_ranks, queries, \
            query_characteristics, visualize=None, iterations=100):
    reit_ = 1e9
    last_reit = 1e9
    SIMILARITY = 1e-5

    for iteration in range(iterations):
        new_population = []
        
        sz = len(population)
        
        new_population.append(population[0])
        
        for i in range(sz // 30):
            new_population.append(mutate_rand_tree(population[0]))
            new_population.append(mutate_rand_tree(population[1]))
            new_population.append(mutate_rand_tree(population[2]))
        for i in range(sz // 10):
            new_population.append(create_random_model(4))
        for i in range(sz):
            t1 = population[np.random.randint(0, sz / 2)]
            t2 = population[np.random.randint(0, sz / 2)]
            new_population.append(mutate_rand_tree(crossfit(t1, t2)))
        
        population = np.array(population + new_population)
        values = rank_population(population, doc_ranks, queries, query_characteristics)
        values = np.array(values)
        
        indexes = np.arange(len(population))
        indexes = sorted(indexes, key=lambda i: -values[i])
        
        population = population[indexes]
        values = values[indexes]
        
        ids = []
        for i in range(20):
            for q in range(i + 1, 20):
                if values[i] - values[q] < SIMILARITY:
                    ids.append(q)
                    
        ids = np.unique(ids)
        population = [population[i] for i in range(100 + len(ids)) if i not in ids]
        
        if visualize is not None:
            visualize(population, values)

        print(-values[0])