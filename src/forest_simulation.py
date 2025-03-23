import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.patches import Circle
import random

from graph_builder import build_graph_from_excel

# Ensure data and results directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Tree species and their properties
tree_species = ['Pine', 'Oak', 'Maple', 'Birch', 'Cedar']
tree_heights = {'Pine': (15, 30), 'Oak': (18, 25), 'Maple': (12, 20), 
               'Birch': (10, 15), 'Cedar': (20, 35)}
tree_colors = {'Pine': '#2F4F2F', 'Oak': '#556B2F', 'Maple': '#8B4513', 
              'Birch': '#A0522D', 'Cedar': '#006400'}

def generate_forest_data(num_trees=30, area_size=100, file_path='data/forest_data.xlsx'):
    """Generate a random forest with various tree species"""
    data = []
    
    for i in range(num_trees):
        species = random.choice(tree_species)
        height_range = tree_heights[species]
        
        # Generate random positions
        pos_x = random.uniform(0, area_size)
        pos_y = random.uniform(0, area_size)
        
        # Generate height based on species typical range
        height = random.uniform(height_range[0], height_range[1])
        
        # Trunk diameter typically correlates with height
        trunk_diameter = height * random.uniform(0.05, 0.1)
        
        # Calculate canopy size based on height
        canopy_size = height * random.uniform(0.3, 0.5)
        
        # Add some randomness to the z-coordinate (ground elevation)
        pos_z = random.uniform(0, 5)
        
        data.append({
            'tree_id': i,
            'tree_species': species,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': pos_z,
            'height': height,
            'trunk_diameter': trunk_diameter,
            'canopy_size': canopy_size
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"✅ Generated forest data with {num_trees} trees and saved to {file_path}")
    return df

def visualize_forest(excel_path='data/forest_data.xlsx', 
                    output_path='results/forest_visualization.png', 
                    show_graph=True,
                    distance_threshold=20.0):
    """Visualize the forest with trees and their connections"""
    # Read forest data
    df = pd.read_excel(excel_path)
    
    # Build the graph
    G = build_graph_from_excel(excel_path, distance_threshold)
    
    # Get positions and other attributes
    positions = {i: (row['pos_x'], row['pos_y']) 
                for i, row in df.iterrows()}
    
    species = df['tree_species'].values
    heights = df['height'].values
    canopy_sizes = df['canopy_size'].values
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw edges first (so they appear behind trees)
    nx.draw_networkx_edges(G, positions, alpha=0.3, width=1.0)
    
    # Draw trees as circles with colors based on species
    for i, (pos, sp, height, canopy) in enumerate(zip(positions.values(), species, heights, canopy_sizes)):
        circle = plt.Circle(pos, radius=canopy/3, 
                          color=tree_colors[sp], 
                          alpha=0.7)
        plt.gca().add_patch(circle)
        
        # Add a small circle for the trunk
        trunk = plt.Circle(pos, radius=canopy/10, 
                          color='#8B4513', 
                          alpha=0.9)
        plt.gca().add_patch(trunk)
    
    # Add labels for tree IDs
    if len(G.nodes) <= 50:  # Only show labels if not too many trees
        labels = {i: str(i) for i in G.nodes()}
        nx.draw_networkx_labels(G, positions, labels, font_size=9, 
                              font_color='black', font_weight='bold')
    
    # Add a legend for tree species
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=species)
                     for species, color in tree_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title and adjust layout
    plt.title("Forest Graph Visualization", fontsize=16)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    
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
    print(f"✅ Forest visualization saved to: {output_path}")
    
    if show_graph:
        plt.show()
    
    return G

if __name__ == "__main__":
    # Generate forest data if it doesn't exist
    excel_path = 'data/forest_data.xlsx'
    if not os.path.exists(excel_path):
        generate_forest_data(num_trees=40, file_path=excel_path)
    
    # Visualize the forest
    G = visualize_forest(excel_path, distance_threshold=25.0)
    
    # Print some network statistics
    print(f"Forest graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Find the tree with the most connections
    degrees = dict(G.degree())
    max_degree_node = max(degrees, key=degrees.get)
    print(f"Tree #{max_degree_node} has the most connections: {degrees[max_degree_node]}") 