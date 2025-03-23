import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist

def build_graph_from_excel(excel_path, distance_threshold=15.0):
    df = pd.read_excel(excel_path)
    positions = df[['pos_x', 'pos_y', 'pos_z']].values
    labels = df['tree_species'].values

    G = nx.Graph()
    for i, (pos, label) in enumerate(zip(positions, labels)):
        G.add_node(i, pos=pos, label=label)

    dists = cdist(positions, positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if dists[i, j] < distance_threshold:
                G.add_edge(i, j, weight=dists[i, j])

    return G

