import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


# خواندن دیتاست Zachary's Karate Club و ساخت گراف
G = nx.karate_club_graph()

# ایجاد ماتریس مجاورت گراف
adjacency_matrix = nx.to_numpy_array(G)

# اجرای Spectral Clustering
n_clusters = 2  # تعداد کلاسترها
spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
labels = spectral_clustering.fit_predict(adjacency_matrix)

# نمایش نتایج
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 4))
plt.subplot(121)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300)
plt.title("Karate Club Graph")

plt.subplot(122)
nx.draw(G, pos, with_labels=True, node_color=labels, cmap=plt.cm.viridis, node_size=300)
plt.title("Spectral Clustering")

plt.show()