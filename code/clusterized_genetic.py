import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import documents_to_vectors
import receive_data
import mutation
import crossover
import create_population
from create_population import create_random_model
from crossover import crossfit
from get_quality import get_quality
from mutation import mutate_rand_tree

import matplotlib.pyplot as plt

def rank_population_clusterized(population, doc_ranks, queries, query_characteristics):   
    qualities = []
    for model in population:
        quality = get_quality(model, doc_ranks, queries, query_characteristics)
        qualities.append(quality)

    return qualities

def vis(population, vals):
    plt.figure(figsize=(10, 5))
    plt.scatter(vals, [pop.get_tokens() for pop in population])
    plt.show()
    print ("!")

def genetic_algo_map(doc_ranks, queries, query_characteristics, iter_cnt=20, values_history=None):
    SIMILARITY = 1e-3
    
    np.random.seed(42)
    size = 40

    population = create_population.create_population(size, 5)

    for iteration in range(iter_cnt):
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
        
        population = np.array(list(population) + new_population)
        values = rank_population_clusterized(population, doc_ranks, queries, query_characteristics)
        values = np.array(values)
        
        indexes = np.arange(len(population))
        indexes = sorted(indexes, key=lambda i: -values[i])
        
        population = population[indexes]
        values = values[indexes]
                    
        ids = np.array([i for i in range(20) if values[i] - values[i + 1] < SIMILARITY])
                    
        population = [population[i] for i in range(len(population)) if i not in ids]
        values = [values[i] for i in range(len(population)) if i not in ids]
        if len(population) > sz:
            population, values = population[:sz], values[:sz]
        
        vis(population, values)
        print(-values[0])
    return population[0]

def superposition_model(x, y, weights, model_list):
    return sum([weights[i] * model_list[i].calc(x, y) for i in range(len(model_list))])

def superposition_predict(X, weights, model_list):
    linear_superposition = lambda x, y: superposition_model(x, y, weights, model_list)
    return np.vectorize(linear_superposition)(X[:, 0], X[:, 1])

class ClusterizedGeneticAlgo:
    def __init__(self, n_clusters=3, superposition_method="lr"):
        self.n_clusters = n_clusters
        self.samples = []
        self.correct_answers = []
        self.model_list = []
        self.coef = np.zeros(n_clusters)
        self.superposition_method = superposition_method
        
    def fit(self, X, y):
        clusterizer = KMeans(n_clusters=self.n_clusters)
        clusterizer.fit(X)
        cluster_nums = clusterizer.predict(X)
        
        if superposition_method == "lr":
            for i in range(self.n_clusters):
                self.samples.append(X[cluster_nums == i])
                self.correct_answers.append(y[cluster_nums == i])

            for (sample, correct_answer) in zip(self.samples, self.correct_answers):
                population = create_population.create_population(30, 5)
                self.model_list.append(genetic_algo_map(sample, correct_answer))

                model_cluster_matr = []

            for (sample, correct_answer) in zip(self.samples, self.correct_answers):
                cluster_results = []

                for model in self.model_list:
                    cluster_results.append(
                        r2_score(correct_answer, np.vectorize(model.calc)(sample[:, 0], sample[:, 1]))
                    )
                model_cluster_matr.append(cluster_results)

            lr_composition = LinearRegression(fit_intercept=False)
            lr_composition.fit(np.array(model_cluster_matr),  np.ones(len(self.samples)))
            self.coef = lr_composition.coef_
        elif superposition_method == "fmin":
            pass
        elif superposition_method == "uniform":
            self.coef = np.ones(self.n_clusters) / float(self.n_clusters)
            
    def predict(self, X):
        return superposition_predict(X, self.coef, self.model_list)
        
