import networkx as nx
from sklearn.manifold import SpectralEmbedding
from node2vec import Node2Vec

# خواندن گراف از فایل یا منبع دیگر
# به عنوان مثال، در اینجا از یک فایل edge_list.txt استفاده شده است
G = nx.karate_club_graph()

# الگوریتم Laplacian Eigenmap
laplacian_matrix = nx.laplacian_matrix(G)
dimensions = 5  # تعداد بعد در نهایت
laplacian_eigenmap = SpectralEmbedding(n_components=dimensions, affinity='nearest_neighbors', eigen_solver='arpack')
laplacian_embedding = laplacian_eigenmap.fit_transform(laplacian_matrix)

# الگوریتم Node2Vec
node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=2)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
node2vec_embedding = {node: model.wv[index] for index, node in enumerate(model.wv.index_to_key)}


# نمایش داده‌های به دست آمده
print("Laplacian Eigenmap Embedding:")
print(laplacian_embedding)

print("\nNode2Vec Embedding:")
print(node2vec_embedding)
