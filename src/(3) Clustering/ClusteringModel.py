from kmeans import kmeans
from dbscan import dbscan
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class ClusteringModel:
    def __init__(self, name, data, encoding, scaling, clustering):
        self.name = name
        self.data = data
        self.encoding = encoding
        self.scaling = scaling
        self.clustering = clustering
        self.labels = None
        self.sil = None
        self.ch = None
        
        if self.clustering == 'k-Means':
            self.labels = self.data.pipe(kmeans)
        elif self.clustering == 'DBSCAN':
            self.labels = self.data.pipe(dbscan)
        
        noise_mask = self.labels != -1
        if len(set(self.labels[noise_mask])) < 2:
            self.sil = None
            self.ch = None
        else:
            self.sil = silhouette_score(self.data[noise_mask], self.labels[noise_mask])
            self.ch = calinski_harabasz_score(self.data[noise_mask], self.labels[noise_mask])