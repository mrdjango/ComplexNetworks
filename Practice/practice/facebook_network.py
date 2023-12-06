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
#     display_name: sci-env
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
from pathlib import Path

# %%
# %matplotlib inline

# %%
facebook = pd.read_csv(
    f"../datasets/facebook_combined.txt.gz",
    compression="gzip",
    sep=" ",
    names=["start_node", "end_node"],
)
G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")
facebook

# %% [markdown]
# ### Visualizing the graph

# %%
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
nx.draw_networkx(G, pos=nx.random_layout(G), ax=ax, **plot_options)

# %%
pos = nx.spring_layout(G, iterations=15, seed=1721)
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)

# %%
colors = ["" for x in range(G.number_of_nodes())]  # initialize colors list
counter = 0
for com in nx.community.label_propagation_communities(G):
    color = "#%06X" % randint(0, 0xFFFFFF)  # creates random RGB color
    counter += 1
    for node in list(
        com
    ):  # fill colors list with the particular color for the community nodes
        colors[node] = color
counter

# %%
plt.figure(figsize=(15, 9))
plt.axis("off")
nx.draw_networkx(
    G, pos=pos, node_size=10, with_labels=False, width=0.15, node_color=colors
)

# %% [markdown]
# ## Basic topological attributes

# %%
print("number_of_nodes: ", G.number_of_nodes())

print("number_of_edges: ", G.number_of_edges())

# %% [markdown]
# ## Ratio cut method

# %%
from networkx.algorithms import community
communities = community.kernighan_lin_bisection(G, max_iter=100)


# %%
pos = nx.spring_layout(G)
nx.draw(G,pos, with_labels=True, node_size =100, node_color='w', node_shape = '.')

for i in range(len(communities)):
    nx.draw_networkx_nodes(G, pos, nodelist=communities[i], node_color=colors[i])

