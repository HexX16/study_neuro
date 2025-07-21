from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,_ = make_blobs(n_samples=300, centers=4, random_state=42)

# Метод локтя (Elbow Method) - подбор оптимального числа класетров 
# Ищем излом ("локоть") — точку, после которой инерция перестаёт быстро снижаться.
# Эта точка — оптимальное число кластеров.
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
# Рисуем график
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Количество кластеров (k)")
plt.ylabel("Инерция (inertia_)")
plt.title("Метод локтя")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.labels_
# Визуализация
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, alpha=0.6)
plt.title("Кластеризация KMeans")
plt.show()