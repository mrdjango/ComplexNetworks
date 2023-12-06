"""
 F-Cmeans Implementaion on Zachary karate club network.
 author: Mustafa Shahbazi Dill
 email: mostafas105@gmail.com
"""

import numpy as np
import skfuzzy as fuzz
from datasets import zachary_dataset
import matplotlib.pyplot as plt

# Define the data points
data = np.array(zachary_dataset())

# Set the number of clusters and fuzziness parameter (m)
n_clusters = 3
m = 2

# Apply FCM clustering
cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(
    data.T, c=n_clusters, m=m, error=0.005, maxiter=1000
)

# U contains the membership values for each data point in each cluster
# cntr contains the cluster centers

# Cluster assignments for each data point
cluster_assignments = np.argmax(U, axis=0)

# Print cluster assignments
print("Cluster Assignments:")
print(cluster_assignments)


# Plot the data points with different colors for each cluster
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    plt.scatter(
        data[cluster_assignments == cluster, 0],
        data[cluster_assignments == cluster, 1],
        label=f"Cluster {cluster+1}",
    )

# Plot cluster centers as 'X'
for center in cntr:
    plt.scatter(center[0], center[1], color="black", marker="x", s=100)

plt.title("Fuzzy C-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
