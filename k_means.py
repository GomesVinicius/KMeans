import math
import random

class KMeans:
    def __init__(self, n_clusters, max_iterations=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.centroids = [[]]

    def fit(self, data):
        self.data = data
        self.centroids = random.sample(data, self.n_clusters)

        for _ in range(self.max_iterations):
            clusters = self._assign_clusters()
            new_centroids = self._calculate_centroids(clusters)

            if self._is_converged(new_centroids):
                break

            self.centroids = new_centroids

    def _assign_clusters(self):
        clusters = [ [] for _ in range(self.n_clusters)]

        for point in self.data:
            centroid_idx = self._closest_centroid(point)
            clusters[centroid_idx].append(point)

        return clusters

    def _closest_centroid(self, point):
        distances = [ math.sqrt( sum( (p-c) ** 2 for p, c in zip(point, centroid) ) ) for centroid in self.centroids ]

        return distances.index( min(distances) )
    
    def _calculate_centroids(self, clusters):
        centroids = []

        for cluster in clusters:
            if cluster:
                centroid = [ sum(dim) / len(cluster) for dim in zip(*cluster) ]
                centroids.append(centroid)
            else:
                centroids.append(random.choice(self.data))

            return centroids
        
    def _is_converged(self, new_centroids):
        for old, new in zip(self.centroids, new_centroids):
            distance = math.sqrt( sum( (old - new) ** 2 for old, new in zip(old, new) ) )

            if distance > self.tol:
                return False
        
        return True

    def predict(self, points):
        return [ self._closest_centroid(point) for point in points ]
        