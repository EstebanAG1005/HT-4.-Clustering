# Sección 2:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from kneed import KneeLocator
from sklearn.metrics import adjusted_rand_score

# Visualizar los datos
data = pd.read_csv("iris.csv")
sns.pairplot(data)
plt.show()


# 2. Crear 2 clusters utilizando KMeans Clustering y graficar los resultados:
kmeans = KMeans(n_clusters=2)
data["cluster"] = kmeans.fit_predict(data)

sns.scatterplot(data=data, x="petal_length", y="petal_width", hue="cluster")
plt.show()

# 3. Estandarizar los datos e intentar el paso 2 de nuevo:
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=["cluster"]))

kmeans = KMeans(n_clusters=2)
data["scaled_cluster"] = kmeans.fit_predict(scaled_data)

sns.scatterplot(data=data, x="petal_length", y="petal_width", hue="scaled_cluster")
plt.show()

# 4. Utilizar el método del "codo" para determinar el número ideal de clusters:
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# 5. Graficar con diferentes números de clusters:
for clusters in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=clusters)
    data[f"scaled_cluster_{clusters}"] = kmeans.fit_predict(scaled_data)

    sns.scatterplot(
        data=data, x="petal_length", y="petal_width", hue=f"scaled_cluster_{clusters}"
    )
    plt.title(f"{clusters} Clusters")
    plt.show()


# Seccion 3
kneedle = KneeLocator(range(1, 11), inertia, curve="convex", direction="decreasing")
print(f"\nNúmero óptimo de clusters según kneed: {kneedle.elbow}\n")

# 6. Comparar con los datos reales:
# Cargar los datos reales y asignar etiquetas numéricas:
real_data = pd.read_csv("iris-con-respuestas.csv")
species_mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
real_data["species_label"] = real_data["species"].map(species_mapping)

# Calcular la matriz de confusión (usando el número óptimo de clusters que hayas encontrado, aquí asumimos 3 clusters)
kmeans = KMeans(n_clusters=3)
data["scaled_cluster"] = kmeans.fit_predict(scaled_data)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(real_data["species_label"], data["scaled_cluster"])


# Calculate the Adjusted Rand Index
ari = adjusted_rand_score(
    real_data["species"], real_data["scaled_cluster"]
)  # Add this line
print(f"Adjusted Rand Index: {ari:.2f}")


# Visualizar la matriz de confusión
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=species_mapping.keys(),
    yticklabels=species_mapping.keys(),
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
