# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
# ---

# %% [markdown]
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# from google.colab import drive
# drive.mount('/content/drive')
# # %cd /content/drive/MyDrive/ColabNotebooks/ComplexNetworks/practice-2


# %%
import networkx as nx
from datasets import zachary_dataset
from pprint import pprint
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import metrics
import numpy as np

# %%
data = zachary_dataset()

# %%
# ایجاد یک گراف (شبکه) جدید
G = nx.Graph()

# %%
G.add_edges_from(data)

# %%
# values_ by Node
values_ = {}
node_to_cluster = {}

# %%
# محاسبه ماتریس مجاورت گراف
adjacency_matrix = nx.to_numpy_array(G)

# %%
# k-path centrality
def k_path_centrality(G, k):
    centrality = {}
    nodes = list(G.nodes())
    for node in nodes:
        total_paths = 0
        for source in nodes:
            if source != node:
                paths = list(
                    nx.all_simple_paths(G, source=source, target=node, cutoff=k)
                )
                total_paths += len(paths)
        centrality[node] = total_paths
    return centrality


# %%
# closeness centrality
closeness_centrality = nx.closeness_centrality(G)

# %%
# betweenness_centrality
betweenness_centrality = nx.betweenness_centrality(G)

# %%
# neighbor_centrality
neighbor_centrality = {}

# %%
# eigenvector_centrality
eigenvector_centrality = nx.eigenvector_centrality(G)

# %%
# k-path centrality
k_path_centrality_ = k_path_centrality(G, k=3)

# %%
# page rank
pagerank = nx.pagerank(G)

# %%
# closeness centrality
for node, centrality in closeness_centrality.items():
    # print(f"Node: {node}: closeness_centrality: {centrality}", end="\n")
    values_[node] = {"closeness_centrality": centrality}

# %%
# betweenness_centrality
for node, centrality in betweenness_centrality.items():
    # print(f"Node: {node}: betweenness_centrality: {centrality}", end="\n")
    values_[node]["betweenness_centrality"] = centrality


# %%
# eigenvector_centrality
for node, centrality in eigenvector_centrality.items():
    # print(f"Node: {node}: eigenvector_centrality: {centrality}", end="\n")
    values_[node]["eigenvector_centrality"] = centrality

# %%
# Neighbor-based centrality
for node in G.nodes():
    neighbor_count = len(list(G.neighbors(node)))
    neighbor_centrality[node] = neighbor_count

# %%
for node, centrality in neighbor_centrality.items():
    # print(f"Node {node}: neighbor_centrality{centrality}")
    values_[node]["neighbor_centrality"] = centrality

# %%
# k_path_centrality
for node, centrality in k_path_centrality_.items():
    # print(f"Node {node}: k_path_centrality: {centrality}")
    values_[node]["k_path_centrality"] = centrality

# %%
# نمایش PageRank برای هر گره
for node, rank in pagerank.items():
    # print(f'گره {node}: PageRank = {rank:.4f}')
    values_[node]["PageRank"] = f"{rank:.4f}"

# %%
# Spectral clustering
num_clusters = 2  # تعداد خوشه‌ها
clustering = SpectralClustering(n_clusters=num_clusters, affinity='rbf', random_state=0)
Spectral_labels = clustering.fit_predict(adjacency_matrix)
## Result 
for node, label in enumerate(Spectral_labels):
    # print(f'گره {node + 1} در خوشه {label + 1}')
    node_to_cluster[node+1] = label + 1 # used for modularity
    values_[node+1]["spectral_cluster"] = label + 1


# %%
# K-Means clustering
## تعداد خوشه‌ها که می‌خواهید تعیین کنید
num_clusters = 2

# %%
## ایجاد یک مدل 
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
## اجرای الگوریتم K-Means بر روی داده
kmeans.fit(data)
## دسته‌بندی داده‌ها به خوشه‌ها
labels = kmeans.labels_
## مراکز خوشه‌ها
cluster_centers = kmeans.cluster_centers_
##  Results
print(len(labels))
for i, label in enumerate(labels):
    # print(f'point {data[i][0]} in cluster: {label + 1}')
    if "kmeans" not in (values_[data[i][0]]):
        values_[data[i][0]]["kmeans"] = [{"point":data[i],"cluster":label + 1}]
    else:
        values_[data[i][0]]["kmeans"] += [{"point":data[i],"cluster":label + 1}]

# %% [markdown]
# Modularity using Spectral Clustering
# # محاسبه معیار Modularity
# print(G.nodes())
# print(node_to_cluster)
# # modularity = nx.community.modularity(G, data, weight='weight')

# %% [markdown]
# ## نمایش معیار Modularity
# print(f'Modularity: {modularity:.4f}')

# %%
pprint(values_)

# %%
