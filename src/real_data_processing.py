import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist
import tensorflow as tf
from PIL import Image
import cv2
import json
import glob
from pathlib import Path

from model_vqa_gat import VQAGATModel

# 确保输出目录存在
os.makedirs("results/real_data", exist_ok=True)

def load_real_forest_data(data_dir, forest_type):
    """加载真实森林数据
    
    参数:
        data_dir (str): 数据根目录路径
        forest_type (str): 森林类型（目录名）
    """
    forest_path = os.path.join(data_dir, forest_type)
    if not os.path.exists(forest_path):
        raise ValueError(f"找不到森林数据目录: {forest_path}")
    
    # 加载对象信息
    obj_info_path = os.path.join(forest_path, "obj_info_final.xlsx")
    if not os.path.exists(obj_info_path):
        raise ValueError(f"找不到对象信息文件: {obj_info_path}")
    
    obj_df = pd.read_excel(obj_info_path)
    print(f"加载了 {len(obj_df)} 个对象的信息")
    
    # 加载景观信息
    landscape_path = os.path.join(forest_path, "landscape_info.txt")
    landscape_info = {}
    if os.path.exists(landscape_path):
        with open(landscape_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    landscape_info[key.strip()] = value.strip()
    
    # 获取RGB图像列表
    rgb_dir = os.path.join(forest_path, "rgb")
    rgb_files = []
    if os.path.exists(rgb_dir):
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    
    # 获取深度图列表
    depth_dir = os.path.join(forest_path, "depth_pfm")
    depth_files = []
    if os.path.exists(depth_dir):
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.pfm")))
    
    # 获取实例分割图列表
    instance_dir = os.path.join(forest_path, "instance_segmentation")
    instance_files = []
    if os.path.exists(instance_dir):
        instance_files = sorted(glob.glob(os.path.join(instance_dir, "*.png")))
    
    return {
        'object_data': obj_df,
        'landscape_info': landscape_info,
        'rgb_files': rgb_files,
        'depth_files': depth_files,
        'instance_files': instance_files
    }

def process_tree_data(obj_df):
    """处理树木数据，提取位置、大小和种类信息"""
    # 初始化树木数据
    tree_data = []
    
    # 查找相关列
    pos_cols = [col for col in obj_df.columns if any(x in col.lower() for x in ['pos', 'position', 'loc', 'x', 'y', 'z'])]
    size_cols = [col for col in obj_df.columns if any(x in col.lower() for x in ['size', 'height', 'width', 'diam', 'crown', 'scale'])]
    type_cols = [col for col in obj_df.columns if any(x in col.lower() for x in ['type', 'species', 'class', 'category', 'material'])]
    
    print(f"找到位置列: {pos_cols}")
    print(f"找到大小列: {size_cols}")
    print(f"找到类型列: {type_cols}")
    
    # 处理每棵树
    for i, row in obj_df.iterrows():
        tree = {'tree_id': i}
        
        # 处理位置
        for col in pos_cols:
            try:
                if 'x' in col.lower():
                    value = str(row[col]).split()[0] if isinstance(row[col], str) else row[col]
                    tree['pos_x'] = float(value)
                elif 'y' in col.lower():
                    value = str(row[col]).split()[0] if isinstance(row[col], str) else row[col]
                    tree['pos_y'] = float(value)
                elif 'z' in col.lower():
                    value = str(row[col]).split()[0] if isinstance(row[col], str) else row[col]
                    tree['pos_z'] = float(value)
            except (ValueError, TypeError):
                continue
        
        # 处理大小
        for col in size_cols:
            try:
                if 'height' in col.lower():
                    value = str(row[col]).split()[0] if isinstance(row[col], str) else row[col]
                    tree['height'] = float(value)
                elif 'crown' in col.lower() or 'diam' in col.lower():
                    value = str(row[col]).split()[0] if isinstance(row[col], str) else row[col]
                    tree['trunk_diameter'] = float(value)
                elif 'scale' in col.lower():
                    value = str(row[col]).split()[0] if isinstance(row[col], str) else row[col]
                    if 'scale_x' not in tree:
                        tree['scale_x'] = float(value)
                    elif 'scale_y' not in tree:
                        tree['scale_y'] = float(value)
                    elif 'scale_z' not in tree:
                        tree['scale_z'] = float(value)
            except (ValueError, TypeError):
                continue
        
        # 处理树种
        for col in type_cols:
            if pd.notna(row[col]):
                value = str(row[col]).lower()
                if 'material' in col.lower():
                    # 将材质索引映射到树种
                    material_to_species = {
                        '0': 'pine',
                        '1': 'oak',
                        '2': 'maple',
                        '3': 'birch',
                        '4': 'cedar',
                        '5': 'redwood',
                        '6': 'palm',
                        '7': 'willow',
                        '8': 'eucalyptus',
                        '9': 'ash'
                    }
                    tree['tree_species'] = material_to_species.get(value, 'unknown')
                else:
                    tree['tree_species'] = value
                break
        
        # 设置默认值和计算派生值
        if 'pos_x' not in tree:
            tree['pos_x'] = 0.0
        if 'pos_y' not in tree:
            tree['pos_y'] = 0.0
        if 'pos_z' not in tree:
            tree['pos_z'] = 0.0
        
        # 使用scale值来估算高度和大小
        if 'scale_z' in tree and 'height' not in tree:
            tree['height'] = tree['scale_z'] * 10.0  # 假设scale_z=1对应10米高
        elif 'height' not in tree:
            tree['height'] = 10.0  # 默认高度
        
        if 'scale_x' in tree and 'trunk_diameter' not in tree:
            tree['trunk_diameter'] = tree['scale_x'] * 0.5  # 假设scale_x=1对应0.5米直径
        elif 'trunk_diameter' not in tree:
            tree['trunk_diameter'] = tree['height'] * 0.1  # 默认为高度的10%
        
        if 'scale_y' in tree:
            tree['canopy_size'] = max(tree['scale_x'], tree['scale_y']) * 5.0  # 使用较大的横向scale
        else:
            tree['canopy_size'] = tree['height'] * 0.3  # 默认为高度的30%
        
        if 'tree_species' not in tree:
            tree['tree_species'] = 'unknown'
        
        tree_data.append(tree)
    
    # 创建DataFrame并返回
    forest_df = pd.DataFrame(tree_data)
    print(f"成功处理了 {len(tree_data)} 棵树的数据")
    return forest_df

def build_forest_graph(forest_df, distance_threshold=20.0, max_edges_per_node=10):
    """根据树木位置构建森林图
    
    参数:
        forest_df: 森林数据DataFrame
        distance_threshold: 连接边的距离阈值
        max_edges_per_node: 每个节点的最大边数
    """
    G = nx.Graph()
    
    # 提取位置信息
    positions = forest_df[['pos_x', 'pos_y', 'pos_z']].values
    
    # 添加节点
    for i, (_, row) in enumerate(forest_df.iterrows()):
        G.add_node(i, 
                   pos=(row['pos_x'], row['pos_y'], row['pos_z']), 
                   species=row['tree_species'],
                   height=row['height'],
                   trunk_diameter=row['trunk_diameter'],
                   canopy_size=row['canopy_size'])
    
    # 分批计算距离和添加边
    batch_size = 1000
    n_nodes = len(positions)
    
    for i in range(0, n_nodes, batch_size):
        batch_end = min(i + batch_size, n_nodes)
        batch_positions = positions[i:batch_end]
        
        # 计算当前批次与所有节点的距离
        distances = cdist(batch_positions, positions)
        
        # 为每个节点添加最近的边
        for j in range(len(batch_positions)):
            node_idx = i + j
            # 获取距离最近的k个节点（不包括自己）
            dist_to_others = distances[j]
            nearest_indices = np.argsort(dist_to_others)[1:max_edges_per_node+1]
            
            # 添加边（只添加距离在阈值内的）
            for other_idx in nearest_indices:
                if dist_to_others[other_idx] < distance_threshold:
                    G.add_edge(node_idx, other_idx, weight=dist_to_others[other_idx])
    
    print(f"创建了一个带有 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的图")
    return G

def visualize_forest_with_gat(G, forest_type, image_path=None):
    """使用GAT模型可视化森林图"""
    # 提取2D位置用于可视化
    node_positions = {i: (G.nodes[i]['pos'][0], G.nodes[i]['pos'][1]) for i in G.nodes()}
    
    # 获取树木属性
    species = [G.nodes[i]['species'] for i in G.nodes()]
    heights = [G.nodes[i]['height'] for i in G.nodes()]
    canopy_sizes = [G.nodes[i]['canopy_size'] for i in G.nodes()]
    
    # 创建画布
    plt.figure(figsize=(16, 12))
    
    # 如果提供了背景图像
    if image_path and os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.imshow(img)
    else:
        # 设置背景颜色
        if 'burned' in forest_type:
            plt.gca().set_facecolor('#E5E5DC')
        elif 'rain' in forest_type:
            plt.gca().set_facecolor('#E0F0E0')
        elif 'redwood' in forest_type:
            plt.gca().set_facecolor('#F0E8DC')
        elif any(x in forest_type for x in ['downtown', 'city', 'suburb']):
            plt.gca().set_facecolor('#F0F0F0')
        else:
            plt.gca().set_facecolor('#F5F5F5')
    
    # 准备GAT输入数据
    A = nx.to_numpy_array(G).astype(np.float32)
    N = len(G.nodes)
    
    # 节点特征
    node_features = []
    for node in G.nodes():
        pos = G.nodes[node]['pos']
        height = G.nodes[node]['height']
        canopy = G.nodes[node]['canopy_size']
        degree = G.degree(node)
        node_features.append(list(pos) + [height, canopy, degree])
    
    node_features = np.array(node_features).astype(np.float32)
    
    # 标准化特征
    mean = np.mean(node_features, axis=0, keepdims=True)
    std = np.std(node_features, axis=0, keepdims=True) + 1e-6
    node_features = (node_features - mean) / std
    
    # 运行GAT模型
    n_classes = len(set(species))
    model = VQAGATModel(n_classes=n_classes, hidden_dim=node_features.shape[1], attn_heads=4)
    
    # 获取注意力权重
    attn_weights = model.get_attention_weights(n_nodes=N)
    
    # 绘制带有注意力权重的边
    for i, j in G.edges():
        plt.plot([node_positions[i][0], node_positions[j][0]], 
                [node_positions[i][1], node_positions[j][1]], 
                color='red', 
                alpha=min(1.0, attn_weights[i, j] * 3), 
                linewidth=attn_weights[i, j] * 5,
                zorder=1)
    
    # 绘制树木
    for i, (pos, sp, height, canopy) in enumerate(zip(node_positions.values(), species, heights, canopy_sizes)):
        # 获取树种颜色
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
    
    # 添加图例
    present_species = list(set(species))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=TREE_COLORS.get(sp, TREE_COLORS['default']), 
                                markersize=10, label=sp.capitalize())
                      for sp in present_species]
    
    plt.legend(handles=legend_elements, loc='upper right', title="树种")
    
    # 设置标题和标签
    plt.title(f"{forest_type.replace('_', ' ').title()} - GAT分析", fontsize=16)
    plt.xlabel("X坐标 (米)")
    plt.ylabel("Y坐标 (米)")
    
    # 保存结果
    output_file = f'results/real_data/{forest_type}_gat_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存GAT分析结果到: {output_file}")
    
    plt.close()

# 树木颜色映射
TREE_COLORS = {
    'pine': '#2F4F2F',      # 深绿色
    'oak': '#556B2F',       # 橄榄绿
    'maple': '#8B4513',     # 棕色
    'birch': '#A0522D',     # 赭石色
    'cedar': '#006400',     # 暗绿色
    'redwood': '#8B2500',   # 深红棕色
    'palm': '#6B8E23',      # 橄榄褐色
    'willow': '#9ACD32',    # 黄绿色
    'eucalyptus': '#228B22',# 森林绿
    'ash': '#808000',       # 橄榄色
    'beech': '#BDB76B',     # 深卡其色
    'fir': '#006400',       # 暗绿色
    'spruce': '#004225',    # 深绿色
    'cypress': '#2E8B57',   # 海绿色
    'unknown': '#3CB371',   # 中绿色
    'default': '#3CB371'    # 中绿色
}

def main():
    """主函数"""
    print("=== 真实森林数据处理与分析 ===")
    
    # 数据根目录
    data_dir = "/Users/yangyiru/Desktop/thesis/DATA"
    
    # 获取所有森林类型
    forest_types = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    print(f"找到 {len(forest_types)} 个森林数据集:")
    for i, forest_type in enumerate(forest_types):
        print(f"{i+1}. {forest_type}")
    
    try:
        # 让用户选择要处理的数据集
        choice = input("\n请选择要处理的数据集编号 (输入 'all' 处理全部，或输入具体编号): ")
        
        if choice.lower() == 'all':
            selected_forests = forest_types
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(forest_types):
                    selected_forests = [forest_types[idx]]
                else:
                    print("无效的选择")
                    return
            except ValueError:
                print("请输入有效的数字")
                return
        
        # 处理选中的数据集
        for forest_type in selected_forests:
            print(f"\n处理数据集: {forest_type}")
            
            try:
                # 加载数据
                forest_data = load_real_forest_data(data_dir, forest_type)
                
                # 处理树木数据
                forest_df = process_tree_data(forest_data['object_data'])
                
                # 构建图 - 限制每个节点的最大边数
                G = build_forest_graph(forest_df, distance_threshold=20.0, max_edges_per_node=10)
                
                # 可视化 - 使用第一张RGB图片作为背景（如果有）
                background_image = forest_data['rgb_files'][0] if forest_data['rgb_files'] else None
                visualize_forest_with_gat(G, forest_type, background_image)
                
                print(f"完成 {forest_type} 的处理\n")
            
            except Exception as e:
                print(f"处理 {forest_type} 时出错: {str(e)}")
                continue
    
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()