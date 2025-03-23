import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
from matplotlib.patches import Circle
from scipy.spatial.distance import cdist
import tensorflow as tf

from model_vqa_gat import VQAGATModel
from graph_builder import build_graph_from_excel

# 确保目录存在
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 数据集列表
FOREST_DATASETS = [
    {
        "name": "meadow_forest",
        "files": ["depth_pfm", "instance_segmentation", "rgb"],
        "semantic_segmentation": True
    },
    {
        "name": "birch_forest",
        "files": ["coco_annotation", "depth_pfm", "instance_segmentation", "landscape_info.txt", 
                 "obj_info_final.xlsx", "rgb", "semantic_segmentation"]
    },
    {
        "name": "broadleaf_forest",
        "files": ["coco_annotation", "depth_pfm", "instance_segmentation", 
                 "landscape_info.txt", "obj_info_final.xlsx", "rgb", "semantic_segmentation"]
    },
    {
        "name": "burned_forest",
        "files": ["coco_annotation", "depth_pfm", "instance_segmentation", "landscape_info.txt",
                 "obj_info_final.xlsx", "rgb", "semantic_segmentation"]
    },
    {
        "name": "city_park",
        "files": ["depth_pfm", "instance_segmentation", "landscape_info.txt", "obj_info_final.xlsx", 
                 "rgb", "semantic_segmentation"]
    },
    {
        "name": "downtown_europe",
        "files": ["depth_pfm", "instance_segmentation", "landscape_info.txt", "obj_info_final.xlsx", 
                 "rgb", "semantic_segmentation"]
    },
    {
        "name": "downtown_west",
        "files": ["depth_pfm", "instance_segmentation", "landscape_info.txt", "obj_info_final.xlsx", 
                 "rgb", "semantic_segmentation"]
    },
    {
        "name": "plantation",
        "files": ["depth_pfm", "instance_segmentation", "landscape_info.txt", "obj_info_final.xlsx", 
                 "rgb", "semantic_segmentation"]
    },
    {
        "name": "rainforest",
        "files": ["coco_annotation", "depth_pfm", "instance_segmentation", "landscape_info.txt", 
                 "obj_info_final.xlsx", "rgb", "semantic_segmentation"]
    },
    {
        "name": "redwood_forest",
        "files": ["coco_annotation", "depth_pfm", "instance_segmentation", "landscape_info.txt", 
                 "obj_info_final.xlsx", "rgb", "semantic_segmentation"]
    },
    {
        "name": "suburb_us",
        "files": ["depth_pfm", "instance_segmentation", "landscape_info.txt", "obj_info_final.xlsx", 
                 "rgb", "semantic_segmentation"]
    }
]

# 树木类型和颜色映射
TREE_COLORS = {
    'pine': '#2F4F2F',
    'oak': '#556B2F', 
    'maple': '#8B4513',
    'birch': '#A0522D',
    'cedar': '#006400',
    'redwood': '#8B2500',
    'palm': '#6B8E23',
    'willow': '#9ACD32',
    'eucalyptus': '#228B22',
    'ash': '#808000',
    'beech': '#BDB76B',
    'fir': '#006400',
    'spruce': '#004225',
    'cypress': '#2E8B57',
    'default': '#3CB371'  # 默认颜色
}

def create_simulated_forest_data(dataset_name, num_trees=50, area_size=100):
    """基于数据集名称创建模拟森林数据"""
    # 根据数据集类型确定树种分布
    species_distribution = {}
    
    if 'birch' in dataset_name:
        species_distribution = {'birch': 0.6, 'pine': 0.2, 'oak': 0.1, 'maple': 0.1}
    elif 'broadleaf' in dataset_name:
        species_distribution = {'oak': 0.4, 'maple': 0.3, 'beech': 0.2, 'ash': 0.1}
    elif 'redwood' in dataset_name:
        species_distribution = {'redwood': 0.7, 'fir': 0.2, 'oak': 0.1}
    elif 'rainforest' in dataset_name:
        species_distribution = {'palm': 0.3, 'eucalyptus': 0.3, 'willow': 0.2, 'cypress': 0.2}
    elif 'plantation' in dataset_name:
        # 种植园通常是单一树种
        species_distribution = {'pine': 0.9, 'oak': 0.1}
    elif 'meadow' in dataset_name:
        species_distribution = {'oak': 0.4, 'birch': 0.3, 'maple': 0.3}
    elif 'downtown' in dataset_name or 'city' in dataset_name or 'suburb' in dataset_name:
        # 城市环境中的树木分布
        species_distribution = {'oak': 0.3, 'maple': 0.3, 'ash': 0.2, 'pine': 0.1, 'palm': 0.1}
    else:  # 默认混合林
        species_distribution = {'pine': 0.25, 'oak': 0.25, 'maple': 0.25, 'birch': 0.25}
    
    # 根据环境调整树木分布模式
    if 'burned' in dataset_name:
        clustering = 'sparse'  # 烧毁的森林树木稀疏
        height_factor = 0.5  # 树木高度降低
    elif 'plantation' in dataset_name:
        clustering = 'grid'  # 种植园呈现规则网格
        height_factor = 1.0
    elif 'rainforest' in dataset_name:
        clustering = 'dense'  # 雨林茂密
        height_factor = 1.3  # 树木较高
    elif 'redwood' in dataset_name:
        clustering = 'medium'
        height_factor = 1.5  # 红木特别高
    elif 'downtown' in dataset_name or 'city' in dataset_name:
        clustering = 'sparse'  # 城市中树木稀少
        height_factor = 0.8
    else:
        clustering = 'medium'
        height_factor = 1.0
    
    # 树高范围（米）
    tree_heights = {
        'pine': (15, 30), 
        'oak': (18, 25), 
        'maple': (12, 20),
        'birch': (10, 15), 
        'cedar': (20, 35),
        'redwood': (30, 60),
        'palm': (10, 25),
        'willow': (15, 25),
        'eucalyptus': (20, 40),
        'ash': (15, 25),
        'beech': (20, 30),
        'fir': (20, 40),
        'spruce': (20, 40),
        'cypress': (15, 25)
    }
    
    # 生成树木数据
    data = []
    
    # 根据聚集模式生成位置
    if clustering == 'grid':
        # 种植园的规则网格排列
        rows = int(np.sqrt(num_trees))
        cols = num_trees // rows + (1 if num_trees % rows > 0 else 0)
        spacing = area_size / max(rows, cols)
        
        for i in range(num_trees):
            row = i // cols
            col = i % cols
            
            # 添加少量随机性
            jitter = spacing * 0.1
            pos_x = col * spacing + random.uniform(-jitter, jitter) + spacing/2
            pos_y = row * spacing + random.uniform(-jitter, jitter) + spacing/2
            
            # 从分布中选择树种
            species = random.choices(
                list(species_distribution.keys()),
                weights=list(species_distribution.values()),
                k=1
            )[0]
            
            # 生成树高
            height_range = tree_heights.get(species, (10, 20))
            height = random.uniform(height_range[0], height_range[1]) * height_factor
            
            # 其他参数
            trunk_diameter = height * random.uniform(0.05, 0.1)
            canopy_size = height * random.uniform(0.3, 0.5)
            pos_z = random.uniform(0, 2)  # 地面起伏
            
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
    
    elif clustering == 'dense' or clustering == 'medium' or clustering == 'sparse':
        # 生成聚集的树木分布
        if clustering == 'dense':
            num_clusters = max(1, num_trees // 10)
            cluster_radius = area_size * 0.15
        elif clustering == 'medium':
            num_clusters = max(1, num_trees // 6)
            cluster_radius = area_size * 0.2
        else:  # sparse
            num_clusters = max(1, num_trees // 4)
            cluster_radius = area_size * 0.25
        
        # 生成簇中心
        cluster_centers = []
        for _ in range(num_clusters):
            cluster_centers.append((
                random.uniform(cluster_radius, area_size - cluster_radius),
                random.uniform(cluster_radius, area_size - cluster_radius)
            ))
        
        for i in range(num_trees):
            # 选择一个簇
            center = random.choice(cluster_centers)
            
            # 在簇周围随机分布
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, cluster_radius)
            pos_x = center[0] + distance * np.cos(angle)
            pos_y = center[1] + distance * np.sin(angle)
            
            # 确保在区域内
            pos_x = min(max(0, pos_x), area_size)
            pos_y = min(max(0, pos_y), area_size)
            
            # 从分布中选择树种
            species = random.choices(
                list(species_distribution.keys()),
                weights=list(species_distribution.values()),
                k=1
            )[0]
            
            # 生成树高
            height_range = tree_heights.get(species, (10, 20))
            height = random.uniform(height_range[0], height_range[1]) * height_factor
            
            # 其他参数
            trunk_diameter = height * random.uniform(0.05, 0.1)
            canopy_size = height * random.uniform(0.3, 0.5)
            pos_z = random.uniform(0, 3 if 'mountain' in dataset_name else 2)
            
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
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到文件
    file_path = f'data/{dataset_name}_simulated.xlsx'
    df.to_excel(file_path, index=False)
    print(f"✅ 已生成{dataset_name}的模拟森林数据 ({num_trees}棵树) 并保存到 {file_path}")
    
    return df, file_path

def visualize_real_forest(dataset_name, distance_threshold=20.0, with_gat=True):
    """可视化真实森林数据集"""
    # 生成模拟数据
    num_trees = random.randint(40, 100 if 'rain' in dataset_name or 'redwood' in dataset_name else 70)
    df, excel_path = create_simulated_forest_data(dataset_name, num_trees=num_trees)
    
    # 使用graph_builder构建图
    G = nx.Graph()
    positions = df[['pos_x', 'pos_y', 'pos_z']].values
    
    # 添加节点
    for i, (pos, row) in enumerate(zip(positions, df.iterrows())):
        G.add_node(i, pos=pos, label=row[1]['tree_species'])
    
    # 添加边 - 基于距离阈值
    dists = cdist(positions, positions)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if dists[i, j] < distance_threshold:
                G.add_edge(i, j, weight=dists[i, j])
    
    # 获取节点位置和其他属性
    node_positions = {i: (row['pos_x'], row['pos_y']) for i, row in df.iterrows()}
    species = df['tree_species'].values
    heights = df['height'].values
    canopy_sizes = df['canopy_size'].values
    
    # 创建图形
    plt.figure(figsize=(16, 12))
    
    # 绘制背景 - 根据森林类型
    if 'burned' in dataset_name:
        plt.gca().set_facecolor('#E5E5DC')  # 灰褐色背景
    elif 'rain' in dataset_name:
        plt.gca().set_facecolor('#E0F0E0')  # 浅绿色背景
    elif 'redwood' in dataset_name or 'pine' in dataset_name:
        plt.gca().set_facecolor('#F0E8DC')  # 棕褐色背景
    elif 'downtown' in dataset_name or 'city' in dataset_name or 'suburb' in dataset_name:
        plt.gca().set_facecolor('#F0F0F0')  # 浅灰色背景 - 城市
    else:
        plt.gca().set_facecolor('#F5F5F5')  # 默认淡色背景
    
    # 如果使用GAT模型
    if with_gat:
        # 准备GAT输入数据
        A = nx.to_numpy_array(G).astype(np.float32)
        
        # 创建简单的节点特征（位置和度）
        node_features = []
        for node in range(len(G.nodes)):
            pos = G.nodes[node].get('pos', [0, 0, 0])
            degree = G.degree(node)
            node_features.append(list(pos) + [degree])
        node_features = np.array(node_features).astype(np.float32)
        
        # 标准化特征
        mean = np.mean(node_features, axis=0, keepdims=True)
        std = np.std(node_features, axis=0, keepdims=True) + 1e-6
        node_features = (node_features - mean) / std
        
        # 创建布局向量
        N = len(G.nodes)
        D = 16  # 布局向量维度
        layout_vector = np.random.rand(N, D).astype(np.float32)
        
        # 转换为张量
        x = tf.convert_to_tensor(node_features)
        a = tf.convert_to_tensor(A)
        layout = tf.convert_to_tensor(layout_vector)
        
        # 运行GAT模型
        model = VQAGATModel(n_classes=len(set(species)), hidden_dim=node_features.shape[1], attn_heads=4)
        output = model([x, a, layout], training=False)
        
        # 获取注意力权重
        attn_weights = model.get_attention_weights(n_nodes=N)
        
        # 绘制带有注意力权重的边
        for i, j in G.edges():
            if attn_weights[i, j] > 0.01:  # 只绘制有注意力的边
                plt.plot([node_positions[i][0], node_positions[j][0]], 
                        [node_positions[i][1], node_positions[j][1]], 
                        color='red', 
                        alpha=min(1.0, attn_weights[i, j] * 3),  # 缩放透明度以提高可见性
                        linewidth=attn_weights[i, j] * 5,  # 根据注意力缩放宽度
                        zorder=1)
    else:
        # 绘制普通边
        nx.draw_networkx_edges(G, node_positions, alpha=0.3, width=1.0)
    
    # 绘制树木
    for i, (pos, sp, height, canopy) in enumerate(zip(node_positions.values(), species, heights, canopy_sizes)):
        # 获取树种颜色，如果没有预定义则使用默认色
        color = TREE_COLORS.get(sp, TREE_COLORS['default'])
        
        # 绘制树冠
        circle = plt.Circle(pos, radius=canopy/3, 
                         color=color, 
                         alpha=0.7, zorder=2)
        plt.gca().add_patch(circle)
        
        # 绘制树干
        trunk = plt.Circle(pos, radius=canopy/10, 
                         color='#8B4513', 
                         alpha=0.9, zorder=3)
        plt.gca().add_patch(trunk)
    
    # 添加节点标签
    if len(G.nodes) <= 60:  # 只在树木不太多时显示标签
        for i, pos in node_positions.items():
            plt.text(pos[0], pos[1], str(i), 
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=8, fontweight='bold',
                   color='white', zorder=4)
    
    # 添加图例 - 仅包含实际出现的树种
    present_species = list(set(species))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=TREE_COLORS.get(sp, TREE_COLORS['default']), 
                              markersize=10, label=sp.capitalize())
                     for sp in present_species]
    
    plt.legend(handles=legend_elements, loc='upper right', title="树种")
    
    # 设置标题
    title = f"{dataset_name.replace('_', ' ').title()} - "
    title += "带GAT注意力分析" if with_gat else "森林图可视化"
    plt.title(title, fontsize=16)
    
    # 网格线
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 设置轴标签
    plt.xlabel("X位置 (米)")
    plt.ylabel("Y位置 (米)")
    
    # 调整轴范围以适应所有树木
    max_x = df['pos_x'].max() + df['canopy_size'].max()
    max_y = df['pos_y'].max() + df['canopy_size'].max()
    min_x = df['pos_x'].min() - df['canopy_size'].max()
    min_y = df['pos_y'].min() - df['canopy_size'].max()
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    
    # 保存可视化
    output_path = f'results/{dataset_name}_{"gat" if with_gat else "regular"}_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ {dataset_name}的{'GAT' if with_gat else '常规'}森林可视化已保存到: {output_path}")
    
    plt.show()
    
    return G

def visualize_all_forests():
    """可视化所有森林数据集"""
    for dataset in FOREST_DATASETS:
        name = dataset["name"]
        print(f"\n正在处理 {name}...")
        
        # 生成两种可视化：常规和GAT
        visualize_real_forest(name, with_gat=False)
        visualize_real_forest(name, with_gat=True)

if __name__ == "__main__":
    # 用户选择模式
    print("请选择要可视化的森林数据集:")
    for i, dataset in enumerate(FOREST_DATASETS):
        print(f"{i+1}. {dataset['name'].replace('_', ' ').title()}")
    print(f"{len(FOREST_DATASETS)+1}. 全部可视化")
    
    try:
        choice = int(input("请输入选项编号: "))
        if choice == len(FOREST_DATASETS) + 1:
            visualize_all_forests()
        elif 1 <= choice <= len(FOREST_DATASETS):
            dataset_name = FOREST_DATASETS[choice-1]["name"]
            print(f"1. 生成普通可视化")
            print(f"2. 生成带GAT注意力的可视化")
            print(f"3. 两种都生成")
            
            viz_choice = int(input("请选择可视化类型: "))
            if viz_choice == 1:
                visualize_real_forest(dataset_name, with_gat=False)
            elif viz_choice == 2:
                visualize_real_forest(dataset_name, with_gat=True)
            elif viz_choice == 3:
                visualize_real_forest(dataset_name, with_gat=False)
                visualize_real_forest(dataset_name, with_gat=True)
        else:
            print("无效选择，将随机选择一个数据集...")
            dataset_name = random.choice(FOREST_DATASETS)["name"]
            visualize_real_forest(dataset_name, with_gat=True)
    except:
        print("输入错误，将自动生成几个示例...")
        # 生成几个代表性案例
        examples = ["birch_forest", "rainforest", "redwood_forest", "city_park"]
        for example in examples:
            visualize_real_forest(example, with_gat=True) 