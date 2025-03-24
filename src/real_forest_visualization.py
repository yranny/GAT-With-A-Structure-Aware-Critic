import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import cdist
import tensorflow as tf

# 导入数据集解析函数
from parse_dataset_info import parse_dataset_info, generate_forest_data
from model_vqa_gat import VQAGATModel

# 确保目录存在
os.makedirs("data", exist_ok=True)
os.makedirs("results/real_forest", exist_ok=True)

# 树木类型和颜色映射
TREE_COLORS = {
    'pine': '#2F4F2F',   # 深绿色
    'oak': '#556B2F',    # 橄榄绿
    'maple': '#8B4513',  # 棕色
    'birch': '#A0522D',  # 赭石色
    'cedar': '#006400',  # 暗绿色
    'redwood': '#8B2500', # 深红棕色
    'palm': '#6B8E23',   # 橄榄褐色
    'willow': '#9ACD32', # 黄绿色
    'eucalyptus': '#228B22', # 森林绿
    'ash': '#808000',    # 橄榄色
    'beech': '#BDB76B',  # 深卡其色
    'fir': '#006400',    # 暗绿色
    'spruce': '#004225', # 深绿色
    'cypress': '#2E8B57', # 海绿色
    'unknown': '#3CB371', # 中绿色
    'default': '#3CB371'  # 中绿色
}

def build_forest_graph(forest_df, distance_threshold=20.0):
    """根据树木位置构建森林图"""
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
    
    # 计算树木之间的距离
    distances = cdist(positions, positions)
    
    # 添加边 - 基于距离阈值
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if distances[i, j] < distance_threshold:
                # 距离作为权重
                G.add_edge(i, j, weight=distances[i, j])
    
    print(f"创建了一个带有 {len(G.nodes)} 个节点和 {len(G.edges)} 条边的图")
    
    return G

def visualize_forest_graph(G, dataset_name, with_gat=True):
    """可视化森林图，可选择使用GAT模型分析"""
    # 提取2D位置用于可视化
    node_positions = {i: (G.nodes[i]['pos'][0], G.nodes[i]['pos'][1]) for i in G.nodes()}
    
    # 获取树木属性
    species = [G.nodes[i]['species'] for i in G.nodes()]
    heights = [G.nodes[i]['height'] for i in G.nodes()]
    canopy_sizes = [G.nodes[i]['canopy_size'] for i in G.nodes()]
    
    # 创建画布
    plt.figure(figsize=(16, 12))
    
    # 设置背景颜色 - 根据森林类型
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
    
    # 如果使用GAT模型分析
    if with_gat:
        # 准备GAT输入数据
        A = nx.to_numpy_array(G).astype(np.float32)
        
        # 节点特征：位置、高度、树冠大小和度
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
        
        # 创建布局向量
        N = len(G.nodes)
        D = 16  # 布局向量维度
        layout_vector = np.random.rand(N, D).astype(np.float32)
        
        # 转换为张量
        x = tf.convert_to_tensor(node_features)
        a = tf.convert_to_tensor(A)
        layout = tf.convert_to_tensor(layout_vector)
        
        # 确定物种类别数量
        n_classes = len(set(species))
        
        # 运行GAT模型
        model = VQAGATModel(n_classes=n_classes, hidden_dim=node_features.shape[1], attn_heads=4)
        output = model([x, a, layout], training=False)
        
        # 获取注意力权重 - 传入节点数量
        attn_weights = model.get_attention_weights(n_nodes=N)
        
        # 绘制带有注意力权重的边
        for i, j in G.edges():
            plt.plot([node_positions[i][0], node_positions[j][0]], 
                    [node_positions[i][1], node_positions[j][1]], 
                    color='red', 
                    alpha=min(1.0, attn_weights[i, j] * 3), 
                    linewidth=attn_weights[i, j] * 5,
                    zorder=1)
    else:
        # 绘制普通边
        nx.draw_networkx_edges(G, node_positions, alpha=0.3, width=1.0, edge_color='gray')
    
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
    
    # 添加节点标签 - 如果节点数量不多
    if len(G.nodes) <= 60:
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
    title += "GAT分析" if with_gat else "常规可视化"
    plt.title(title, fontsize=16)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 设置轴标签
    plt.xlabel("X坐标 (米)")
    plt.ylabel("Y坐标 (米)")
    
    # 调整坐标轴范围
    x_coords = [pos[0] for pos in node_positions.values()]
    y_coords = [pos[1] for pos in node_positions.values()]
    max_canopy = max(canopy_sizes)
    
    plt.xlim(min(x_coords) - max_canopy, max(x_coords) + max_canopy)
    plt.ylim(min(y_coords) - max_canopy, max(y_coords) + max_canopy)
    
    # 保存图像
    output_file = f'results/real_forest/{dataset_name}_{"gat" if with_gat else "regular"}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ {dataset_name}的{'GAT分析' if with_gat else '常规可视化'}已保存到: {output_file}")
    
    plt.close()

def process_real_data(dataset_text):
    """处理数据集信息文本并生成可视化"""
    # 解析数据集信息
    datasets = parse_dataset_info(dataset_text)
    
    if not datasets:
        print("没有找到有效的数据集信息。")
        return
    
    print(f"找到 {len(datasets)} 个数据集:")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset['name']}")
    
    # 处理每个数据集
    for dataset in datasets:
        dataset_name = dataset['name']
        
        # 生成森林数据
        forest_df, _ = generate_forest_data(dataset)
        
        # 构建图
        G = build_forest_graph(forest_df)
        
        # 可视化 - 常规
        visualize_forest_graph(G, dataset_name, with_gat=False)
        
        # 可视化 - GAT分析
        visualize_forest_graph(G, dataset_name, with_gat=True)
        
        print(f"完成 {dataset_name} 的处理\n")

def main():
    print("=== 真实森林数据可视化 ===")
    
    # 默认数据集文本
    default_text = """
"meadow_forest": ["depth_pfm", "instance_segmentation", "rgb",
"semantic_segmentation"],
"birch_forest", coco_annotation, depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"broadleaf_forest", coco_annotation, depth_pfm, instance_segmentation,
landscape_info.txt, obj_info_final.xlsx, rgb, semantic_segmentation
"burned_forest", coco_annotation, depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"city_park", depth_pfm, instance_segmentation, landscape_info.txt, obj_info_final.xlsx, 
rgb, semantic_segmentation
"downtown_europe", depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"downtown_west", depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"plantation", depth_pfm, instance_segmentation, landscape_info.txt, obj_info_final.xlsx,
rgb, semantic_segmentation
"rainforest", coco_annotation, depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"redwood_forest", coco_annotation, depth_pfm, instance_segmentation,
landscape_info.txt, obj_info_final.xlsx, rgb, semantic_segmentation
"suburb_us", depth_pfm, instance_segmentation, landscape_info.txt, obj_info_final.xlsx,
rgb, semantic_segmentation
"""
    
    # 从文件获取数据
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r') as f:
                dataset_text = f.read()
            process_real_data(dataset_text)
        except Exception as e:
            print(f"读取文件时出错: {e}")
            print("将使用默认数据...")
            process_real_data(default_text)
    else:
        # 尝试从用户输入获取数据
        print("请输入数据集信息，或直接按回车使用默认数据集：")
        user_input = input()
        
        if user_input.strip():
            process_real_data(user_input)
        else:
            print("使用默认数据集...")
            process_real_data(default_text)

if __name__ == "__main__":
    main() 