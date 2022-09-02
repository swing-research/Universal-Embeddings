import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt; plt.ion()

"""
Generate tree.

INPUT:
 -Nlevel: number of tree level.
 -Nrep: number of leaves per node.
 -seed: seed parameter.
"""
def tree(Nlevel=6,Nrep=2,seed=42):
    np.random.seed(seed)

    # Construct tree
    G = nx.Graph()
    G.add_node(0)
    node_crt = 0
    node_prev = [0]
    index_list = [0]
    for i in range(Nlevel):
        node_prev_ = []
        while len(node_prev_)==0:
            for j in node_prev:
                for k in range(Nrep):
                    node_crt +=1 
                    G.add_node(node_crt)
                    G.add_edge(j, node_crt, weight=1)
                    node_prev_.append(node_crt)
                    index_list.append(node_crt)
        node_prev = node_prev_

    # Compute distance
    Npts = len(G.nodes)
    dist_tree = np.zeros((Npts,Npts))
    index_list = np.random.choice(index_list,len(index_list),replace=False)
    for i in tqdm(range(index_list.shape[0])):
        for j in range(index_list.shape[0]):
            dist_tree[i,j] = nx.dijkstra_path_length(G,index_list[i],index_list[j])
    dist_tree /= dist_tree.max()

    return G, dist_tree, index_list
