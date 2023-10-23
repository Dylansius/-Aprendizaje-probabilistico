# Datos de ejemplo
data = [
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
]

# Parámetros iniciales
k = 2  # Número de clusters
max_iter = 100  # Máximo de iteraciones

# Inicialización de centroides de manera aleatoria
import random
random.seed(0)
centroids = random.sample(data, k)

# Función para asignar puntos a los clusters
def asignar_a_clusters(data, centroids):
    cluster_assignments = []
    for point in data:
        distances = [sum((x - y) ** 2 for x, y in zip(point, centroid)) ** 0.5 for centroid in centroids]
        cluster_assignments.append(distances.index(min(distances)))
    return cluster_assignments

# Algoritmo K-Means
for _ in range(max_iter):
    cluster_assignments = asignar_a_clusters(data, centroids)
    
    # Actualización de los centroides
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if cluster_assignments[j] == i]
        if cluster_points:
            new_centroid = [sum(point[j] for point in cluster_points) / len(cluster_points) for j in range(len(cluster_points[0]))]
            centroids[i] = new_centroid

# Resultados
for i, centroid in enumerate(centroids):
    print(f"Cluster {i + 1} - Centroide: {centroid}")

for i in range(k):
    cluster_points = [data[j] for j in range(len(data)) if cluster_assignments[j] == i]
    print(f"Puntos en el Cluster {i + 1}:\n{cluster_points}")

