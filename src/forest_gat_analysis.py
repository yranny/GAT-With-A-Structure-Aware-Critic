import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import tensorflow as tf
from scipy.spatial.distance import cdist

from model_vqa_gat import VQAGATModel
from graph_builder import build_graph_from_excel
from forest_simulation import visualize_forest, generate_forest_data

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

def prepare_graph_data(graph):
    """Convert NetworkX graph to adjacency matrix and node features for GAT"""
    # Get number of nodes
    N = len(graph.nodes)
    
    # Create adjacency matrix
    A = nx.to_numpy_array(graph).astype(np.float32)
    
    # Create simple node features (we'll use position and degree as features)
    node_attrs = {}
    for node in graph.nodes():
        # Get position attributes
        pos = graph.nodes[node].get('pos', [0, 0, 0])
        # Degree centrality
        degree = graph.degree(node)
        # Combine features
        node_attrs[node] = list(pos) + [degree]
    
    # Convert to feature matrix
    node_features = np.array([node_attrs[i] for i in range(N)]).astype(np.float32)
    
    # Normalize features
    mean = np.mean(node_features, axis=0, keepdims=True)
    std = np.std(node_features, axis=0, keepdims=True) + 1e-6
    node_features = (node_features - mean) / std
    
    return A, node_features

def visualize_gat_forest(graph, attention_weights, excel_path, 
                         output_path='results/forest_gat_analysis.png'):
    """Visualize the forest with GAT attention weights"""
    # Read forest data
    df = pd.read_excel(excel_path)
    
    # Get positions and other attributes
    positions = {i: (row['pos_x'], row['pos_y']) 
                for i, row in df.iterrows()}
    
    species = df['tree_species'].values
    canopy_sizes = df['canopy_size'].values
    
    # Define tree colors based on species
    tree_colors = {'Pine': '#2F4F2F', 'Oak': '#556B2F', 'Maple': '#8B4513', 
                  'Birch': '#A0522D', 'Cedar': '#006400'}
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Draw edges with attention weights
    for i, j in graph.edges():
        if attention_weights[i, j] > 0.01:  # Only draw edges with attention
            plt.plot([positions[i][0], positions[j][0]], 
                    [positions[i][1], positions[j][1]], 
                    color='red', 
                    alpha=min(1.0, attention_weights[i, j] * 2),  # Scale alpha for visibility
                    linewidth=attention_weights[i, j] * 5,  # Scale width by attention
                    zorder=1)
    
    # Draw trees as circles with colors based on species
    for i, (pos, sp, canopy) in enumerate(zip(positions.values(), species, canopy_sizes)):
        circle = plt.Circle(pos, radius=canopy/3, 
                          color=tree_colors.get(sp, '#008000'), 
                          alpha=0.7, zorder=2)
        plt.gca().add_patch(circle)
        
        # Add a small circle for the trunk
        trunk = plt.Circle(pos, radius=canopy/10, 
                          color='#8B4513', 
                          alpha=0.9, zorder=3)
        plt.gca().add_patch(trunk)
    
    # Add labels for tree IDs
    if len(graph.nodes) <= 50:  # Only show labels if not too many trees
        for i, pos in positions.items():
            plt.text(pos[0], pos[1], str(i), 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=9, fontweight='bold',
                    color='white', zorder=4)
    
    # Add a legend for tree species
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=tree_colors.get(sp, '#008000'), 
                                markersize=10, label=sp)
                      for sp in set(species)]
    plt.legend(handles=legend_elements, loc='upper right', title="Tree Species")
    
    # Set title and adjust layout
    plt.title("Forest Graph with GAT Attention Analysis", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Set axis labels
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    
    # Adjust axis limits to fit all trees with some margin
    max_x = df['pos_x'].max() + df['canopy_size'].max()
    max_y = df['pos_y'].max() + df['canopy_size'].max()
    min_x = df['pos_x'].min() - df['canopy_size'].max()
    min_y = df['pos_y'].min() - df['canopy_size'].max()
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ GAT Forest analysis visualization saved to: {output_path}")
    
    plt.show()

def run_gat_analysis():
    """Run GAT model on forest data and visualize attention weights"""
    # Prepare forest data if needed
    excel_path = 'data/forest_data.xlsx'
    if not os.path.exists(excel_path):
        generate_forest_data(num_trees=40, file_path=excel_path)
    
    # Build the graph
    graph = build_graph_from_excel(excel_path, distance_threshold=25.0)
    
    # Prepare data for GAT
    adjacency, node_features = prepare_graph_data(graph)
    
    # Print graph info
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"Node features shape: {node_features.shape}")
    
    # Create mock layout vector (symbolic structure information)
    F = node_features.shape[1]  # Feature dimension
    D = 16  # Layout vector dimension
    N = len(graph.nodes)
    layout_vector = np.random.rand(N, D).astype(np.float32)
    
    # Convert to tensors
    x = tf.convert_to_tensor(node_features)
    a = tf.convert_to_tensor(adjacency)
    layout = tf.convert_to_tensor(layout_vector)
    
    # Define and run the model
    n_classes = 5  # Number of tree species
    model = VQAGATModel(n_classes=n_classes, hidden_dim=F, attn_heads=4)
    output = model([x, a, layout], training=False)
    
    # Get attention weights with correct node count
    attn_weights = model.get_attention_weights(n_nodes=N)
    
    # Visualize the forest with attention weights
    visualize_gat_forest(graph, attn_weights, excel_path)
    
    return graph, attn_weights

if __name__ == "__main__":
    graph, attention = run_gat_analysis() 