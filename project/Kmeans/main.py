import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def init_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for x in X: 
        distances = [euclidean_distance(x,c) for c in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(x)
    return clusters

def update_centroids(clusters):
    return np.array([np.mean(cluster, axis=0) if len(cluster)>0 else np.zeros(2) for cluster in clusters])

def has_converged(old_centroid, new_centroid):
    return np.allclose(old_centroid, new_centroid)

def kmeans(X, k, max_iters = 100):
    centroids = init_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(clusters)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids,clusters



#Запуск:
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [8.0, 8.0], [1.0, 0.6], [9.0, 11.0],
    [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
])

centroids, clusters = kmeans(X, k=3)

colors = ['r', 'g', 'b']
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f"Cluster {i}")

plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=150, label="Centroids")

plt.title("KMeans Result")
plt.legend()
plt.grid(True)
plt.show()