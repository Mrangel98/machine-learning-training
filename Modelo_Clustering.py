import pandas
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

#Carga del dataset iris
iris = load_iris()

#caracteristicas sin etiquetas

x = iris.data
y_real = iris.target

print ("=== Formas del dataset ===")
print (f"Filas: {x.shape[0]}, Columnas: {x.shape[1]}")

print ("\n=== Primeras filas del dataset ===")
print(iris.feature_names)
print (x[:5])

#Estandarización

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print ("\n=== Datos originales ===")
print (x[:3])
print ("\n=== Datos estandarizados ===")
print (x_scaled[:3])

#entrenamos de KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_scaled)

#obtenemos clusters
clusters = kmeans.labels_
print("=== Clusters asignados a cada flor ===")
print(clusters)

print("\n=== Cuántas flores hay en cada cluster ===")
import numpy as np
unique, counts = np.unique(clusters, return_counts=True)
print(dict(zip(unique, counts)))


# Reducimos de 4 dimensiones a 2 para poder visualizar
pca = PCA(n_components=2)
X_2d = pca.fit_transform(x_scaled)

# Creamos la gráfica
plt.figure(figsize=(12, 5))

# Gráfica 1: Clusters de K-Means
plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusters, cmap='viridis')
plt.title("Clusters K-Means")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Cluster")

# Gráfica 2: Clases reales
plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_real, cmap='viridis')
plt.title("Clases reales")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.colorbar(label="Especie")

plt.tight_layout()
plt.show()

import pandas as pd

# Creamos un DataFrame comparando clusters vs clases reales
comparacion = pd.DataFrame({
    "Cluster KMeans": clusters,
    "Especie Real": y_real,
    "Nombre Especie": [iris.target_names[i] for i in y_real]
})

print("=== Comparación clusters vs especies reales ===")
print(comparacion.groupby(["Cluster KMeans", "Nombre Especie"]).size())