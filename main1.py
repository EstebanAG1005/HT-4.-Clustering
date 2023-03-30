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
from scipy.optimize import linear_sum_assignment


# Visualizar los datos
data = pd.read_csv("iris.csv")
sns.pairplot(data)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sin escalar
kmeans = KMeans(n_clusters=2, n_init=10)
data["cluster"] = kmeans.fit_predict(data)
centroids = kmeans.cluster_centers_

sns.scatterplot(ax=axes[0], data=data, x="sepal_length", y="sepal_width", hue="cluster")
axes[0].scatter(
    centroids[:, 0], centroids[:, 1], marker="x", s=100, c="red", label="Centroids"
)
axes[0].set_title("Without Scaling")
axes[0].legend()

# Con escalamiento
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(columns=["cluster"]))
kmeans = KMeans(n_clusters=2, n_init=10)
data["scaled_cluster"] = kmeans.fit_predict(scaled_data)
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)

sns.scatterplot(
    ax=axes[1], data=data, x="sepal_length", y="sepal_width", hue="scaled_cluster"
)
axes[1].scatter(
    centroids[:, 0], centroids[:, 1], marker="x", s=100, c="red", label="Centroids"
)
axes[1].set_title("With Scaling")
axes[1].legend()

plt.show()


# Utilizar el método del "codo" para determinar el número ideal de clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Graficar con diferentes números de clusters
for clusters in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=clusters, n_init=10)
    data[f"scaled_cluster_{clusters}"] = kmeans.fit_predict(scaled_data)

    centroids = kmeans.cluster_centers_  # Get the centroids
    centroids = scaler.inverse_transform(
        centroids
    )  # Inverse transform the centroids to the original scale

    sns.scatterplot(
        data=data, x="sepal_length", y="sepal_width", hue=f"scaled_cluster_{clusters}"
    )
    plt.scatter(  # Add this block to plot the centroids
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=100,
        c="red",
        label="Centroids",
    )
    plt.title(f"{clusters} Clusters")
    plt.legend()  # Add this line to show the legend
    plt.show()

# Seccion 3
kneedle = KneeLocator(range(1, 11), inertia, curve="convex", direction="decreasing")
print(f"\nNúmero óptimo de clusters según kneed: {kneedle.elbow}\n")

# Comparar con los datos reales
real_data = pd.read_csv("iris-con-respuestas.csv")

species_mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
real_data["species"] = real_data["species"].map(species_mapping)

# Comparar los clusters con las etiquetas reales
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(data)
pred = kmeans.predict(data)

print(pd.Series(pred).value_counts())

scaled_data = scaler.fit_transform(real_data.drop(columns=["species"]))
real_data["scaled_cluster"] = kmeans.fit_predict(scaled_data)

# Match cluster labels to true labels
def match_labels(y_true, y_pred):
    cost_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cost_matrix[i, j] = -1 * np.sum((y_true == i) & (y_pred == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_mapping = dict(zip(col_ind, row_ind))
    return np.array([label_mapping[i] for i in y_pred])


# Apply the label matching
matched_labels = match_labels(real_data["species"], real_data["scaled_cluster"])

# Calculate the accuracy
accuracy = np.sum(matched_labels == real_data["species"]) / len(real_data["species"])
print(f"Accuracy: {accuracy:.2f}")

# Calcular la matriz de confusión
cm = confusion_matrix(real_data["species"], real_data["scaled_cluster"])
classes = unique_labels(real_data["species"], real_data["scaled_cluster"])

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
    xticklabels=classes,
    yticklabels=classes,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
