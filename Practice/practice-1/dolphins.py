#Import necessary libraries:
from pathlib import Path
import networkx as nx
import os
#Graphing:
import matplotlib.pylab as plt
#To be used for converting files GML to JSON:
# import simplejson as json
import json
from networkx.readwrite import json_graph

g = nx.read_gml(f'{Path(__file__).parent.as_posix()}/dolphins.gml')




print('average_clustering: ', nx.average_clustering(g))

print('average_shortest_path_length: ', nx.average_shortest_path_length(g))


# nx.draw(g)
# plt.show()

nx.draw_circular(g,with_labels=True)
plt.show()


