import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# Cargar los datos
data = pd.read_csv("iris.csv")

# Estandarizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Crear 2 clusters utilizando KMeans Clustering y graficar los resultados
kmeans = KMeans(n_clusters=2)
data["cluster"] = kmeans.fit_predict(data)

sns.scatterplot(data=data, x="sepal_length", y="sepal_width", hue="cluster")
plt.show()

# Estandarizar los datos e intentar el paso 2 de nuevo
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=["cluster"]))

kmeans = KMeans(n_clusters=2)
data["scaled_cluster"] = kmeans.fit_predict(scaled_data)

sns.scatterplot(data=data, x="sepal_length", y="sepal_width", hue="scaled_cluster")
plt.show()

# Utilizar el método del "codo" para determinar el número ideal de clusters
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

# Graficar con diferentes números de clusters
for clusters in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=clusters)
    data[f"scaled_cluster_{clusters}"] = kmeans.fit_predict(scaled_data)

    sns.scatterplot(
        data=data, x="sepal_length", y="sepal_width", hue=f"scaled_cluster_{clusters}"
    )
    plt.title(f"{clusters} Clusters")
    plt.show()

# Comparar con los datos reales
real_data = pd.read_csv("iris-con-respuestas.csv")

species_mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
real_data["species"] = real_data["species"].map(species_mapping)

# Comparar los clusters con las etiquetas reales
kmeans = KMeans(n_clusters=3)
scaled_data = scaler.fit_transform(real_data.drop(columns=["species"]))
real_data["scaled_cluster"] = kmeans.fit_predict(scaled_data)

# Calcular la matriz de confusión
cm = confusion_matrix(real_data["species"], real_data["scaled_cluster"])
classes = unique_labels(real_data["species"], real_data["scaled_cluster"])

# Visualizar la matriz de confusión
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
