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
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# %%
# تبدیل گراف به بردارهای نمایشی
def graph_to_vectors(graph):
    adj_matrix = nx.to_numpy_array(graph)
    adj_matrix = torch.FloatTensor(adj_matrix)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = F.relu(GraphConv(2, 16)(Data(x=adj_matrix, edge_index=edge_index)))
    x = F.relu(GraphConv(16, 32)(Data(x=x)))
    x = F.relu(GraphConv(32, 2)(Data(x=x)))
    return x

# %%
num_classes = 2
# کلاس بند
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(2, num_classes)  # num_classes باید تعریف شده باشد

    def forward(self, x):
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# %%
# تعریف گراف KarateClub
karate_club = nx.karate_club_graph()


# %%
# تبدیل گراف به بردارهای نمایشی
def graph_to_vectors(graph):
    adj_matrix = nx.to_numpy_array(graph)
    adj_matrix = torch.FloatTensor(adj_matrix)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    x = F.relu(GraphConv(34, 16)(x=adj_matrix, edge_index=edge_index))  # Adjust the input dimension to 34
    x = F.relu(GraphConv(16, 32)(x=x, edge_index=edge_index))
    x = F.relu(GraphConv(32, 2)(x=x, edge_index=edge_index))
    return x
vectors = graph_to_vectors(karate_club)

# %%
# تعریف کلاس‌ها
labels = [karate_club.nodes[i]['club'] for i in karate_club.nodes]

# %%
# تعریف مدل
num_classes = len(set(labels))
model = Classifier()

# %%
# تعریف مسائل بهینه‌سازی و ترین مدل
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %%
# ترین مدل
for epoch in range(100):
    optimizer.zero_grad()
    output = model(vectors)
    labels_tensor = torch.LongTensor(labels)  # Convert labels to a tensor
    loss = criterion(output, labels_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')



# %%
# ارزیابی مدل
with torch.no_grad():
    model.eval()
    predictions = model(vectors).argmax(dim=1)
    accuracy = torch.eq(predictions, torch.LongTensor(labels)).sum().item() / len(labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')
