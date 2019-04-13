import numpy as np
from sklearn.cluster import KMeans

class Clusterize:
    def __init__(self):
        self.clusterizer = KMeans(random_state=42)
        
    def __init__(self, n_clusters):
        self.clusterizer = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit(self, document_characteristics):
        self.clusterizer.fit(document_characteristics)
        
    def predict(self, document_characteristics):
        return self.predict(document_characteristics)