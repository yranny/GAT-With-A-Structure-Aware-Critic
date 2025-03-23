import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import networkx as nx
import os

from model_vqa_gat import VQAGATModel

# ✅ Ensure result folder exists
os.makedirs("results", exist_ok=True)

# 1. Create toy graph data
N = 6      # Number of nodes
F = 16     # Node feature dimension
D = 16     # Layout vector dimension
n_classes = 4

np.random.seed(42)
x = np.random.rand(N, F).astype(np.float32)
layout_vector = np.random.rand(N, D).astype(np.float32)

# Simple symmetric adjacency matrix
edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
A = np.zeros((N, N), dtype=np.float32)
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1
a = tf.convert_to_tensor(A)

# 2. Build and run model
model = VQAGATModel(n_classes=n_classes, hidden_dim=F, attn_heads=4)
output = model([x, a, layout_vector], training=False)
attn = model.get_attention_weights().numpy()

print("Output shape:", output.shape)
print("Sample output (softmax):", output.numpy()[0])

# 3. Visualize attention as graph
def visualize_attention(adj_matrix, attn_matrix, save_path="results/attention_heatmap.png"):
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=600)
    nx.draw_networkx_labels(G, pos)

    # Draw attention-weighted edges
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] > 0 and attn_matrix[i, j] > 0.01:
                alpha = attn_matrix[i, j]
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(i, j)],
                    width=3 * alpha,
                    edge_color='red',
                    alpha=alpha
                )

    plt.title("GAT Attention Heatmap")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Attention heatmap saved to: {save_path}")
    plt.show()

# Average attention over heads if needed
if attn.ndim == 3:
    attn = attn.mean(axis=0)

visualize_attention(A, attn)

plt.savefig("results/attention_heatmap.png")
plt.show()
