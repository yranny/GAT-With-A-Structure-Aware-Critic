import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# 确保目录存在
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

def parse_dataset_info(text):
    """
    解析数据集信息文本，提取数据集名称和文件列表
    
    参数:
        text (str): 包含数据集信息的文本
        
    返回:
        list: 数据集字典列表，每个字典包含名称和属性
    """
    datasets = []
    
    # 用正则表达式找出数据集名称和文件
    pattern = r'"([^"]+)":|"([^"]+)",|\b([a-zA-Z_]+_[a-zA-Z_]+)\b'
    
    current_dataset = None
    file_list = []
    
    lines = text.strip().split('\n')
    for line in lines:
        # 查找匹配
        matches = re.finditer(pattern, line)
        
        for match in matches:
            # 如果是数据集名称（带引号或不带引号）
            dataset_name = match.group(1) or match.group(2) or match.group(3)
            
            # 如果找到新数据集名称
            if dataset_name and ('forest' in dataset_name or 'park' in dataset_name 
                                or 'downtown' in dataset_name or 'suburb' in dataset_name 
                                or 'plantation' in dataset_name or 'rainforest' in dataset_name):
                
                # 保存之前的数据集
                if current_dataset:
                    datasets.append({
                        'name': current_dataset,
                        'files': file_list
                    })
                
                # 开始新数据集
                current_dataset = dataset_name
                file_list = []
            # 否则是文件名
            elif dataset_name and current_dataset:
                file_list.append(dataset_name)
    
    # 添加最后一个数据集
    if current_dataset and file_list:
        datasets.append({
            'name': current_dataset,
            'files': file_list
        })
    
    # 分析每个数据集并添加属性
    for dataset in datasets:
        # 基于数据集名称设置基本属性
        name = dataset['name']
        
        # 设置树木密度
        if 'city' in name or 'downtown' in name or 'suburb' in name:
            dataset['tree_density'] = 'sparse'
        elif 'burned' in name:
            dataset['tree_density'] = 'very_sparse'
        elif 'rain' in name or 'redwood' in name:
            dataset['tree_density'] = 'very_dense'
        else:
            dataset['tree_density'] = 'medium'
        
        # 设置区域大小 (米)
        if 'downtown' in name or 'city' in name:
            dataset['area_size'] = (200, 200)
        elif 'suburb' in name:
            dataset['area_size'] = (300, 300)
        else:
            dataset['area_size'] = (400, 400)
        
        # 设置主要树种
        if 'birch' in name:
            dataset['primary_species'] = ['birch', 'oak', 'maple']
        elif 'redwood' in name:
            dataset['primary_species'] = ['redwood', 'fir', 'cedar']
        elif 'rainforest' in name:
            dataset['primary_species'] = ['palm', 'eucalyptus']
        elif 'burned' in name:
            dataset['primary_species'] = ['pine', 'burned_pine', 'burned_oak']
        elif 'city' in name or 'downtown' in name or 'suburb' in name:
            dataset['primary_species'] = ['oak', 'maple', 'ash', 'birch']
        else:
            dataset['primary_species'] = ['pine', 'oak', 'maple', 'birch']
        
        # 设置树高范围
        if 'redwood' in name:
            dataset['height_range'] = (20, 80)
        elif 'rainforest' in name:
            dataset['height_range'] = (10, 40)
        elif 'city' in name or 'downtown' in name or 'suburb' in name:
            dataset['height_range'] = (5, 20)
        else:
            dataset['height_range'] = (8, 30)
        
        # 设置树冠大小范围
        if 'redwood' in name:
            dataset['canopy_range'] = (5, 15)
        elif 'rainforest' in name:
            dataset['canopy_range'] = (6, 20)
        elif 'city' in name or 'downtown' in name or 'park' in name:
            dataset['canopy_range'] = (3, 10)
        else:
            dataset['canopy_range'] = (4, 12)
            
        # 附加文件信息
        files = dataset['files']
        if 'obj_info_final.xlsx' in files:
            dataset['has_tree_data'] = True
        else:
            dataset['has_tree_data'] = False
            
        if 'landscape_info.txt' in files:
            dataset['has_landscape_info'] = True
        else:
            dataset['has_landscape_info'] = False
    
    return datasets

def generate_forest_data(dataset_info):
    """
    根据数据集信息生成模拟森林数据
    
    参数:
        dataset_info (dict): 数据集信息字典
        
    返回:
        pd.DataFrame: 树木数据表
        dict: 森林参数
    """
    name = dataset_info['name']
    area_size = dataset_info['area_size']
    primary_species = dataset_info['primary_species']
    height_range = dataset_info['height_range']
    canopy_range = dataset_info['canopy_range']
    tree_density = dataset_info['tree_density']
    
    # 基于密度确定树木数量
    if tree_density == 'very_sparse':
        num_trees = int((area_size[0] * area_size[1]) / 4000)
    elif tree_density == 'sparse':
        num_trees = int((area_size[0] * area_size[1]) / 2000)
    elif tree_density == 'medium':
        num_trees = int((area_size[0] * area_size[1]) / 1000)
    elif tree_density == 'dense':
        num_trees = int((area_size[0] * area_size[1]) / 500)
    else:  # very_dense
        num_trees = int((area_size[0] * area_size[1]) / 200)
    
    # 确保至少有10棵树
    num_trees = max(10, num_trees)
    
    # 生成树木数据
    tree_data = []
    for i in range(num_trees):
        # 树种 - 主要是主要树种，小概率是其他树种
        if random.random() < 0.85:
            species = random.choice(primary_species)
        else:
            # 其他树种
            other_species = ['pine', 'oak', 'maple', 'birch', 'cedar', 'fir', 
                            'spruce', 'cypress', 'ash', 'beech', 'willow']
            # 移除主要树种，避免重复
            other_species = [s for s in other_species if s not in primary_species]
            species = random.choice(other_species)
        
        # 位置 - 添加一些随机聚集
        cluster_center = None
        if random.random() < 0.7:  # 70%的概率在聚集区域
            # 选择或创建聚集中心
            if tree_data and random.random() < 0.8:
                # 选择现有树作为聚集中心
                cluster_tree = random.choice(tree_data)
                cluster_center = (cluster_tree['pos_x'], cluster_tree['pos_y'])
            else:
                # 创建新的聚集中心
                cluster_center = (
                    random.uniform(0, area_size[0]),
                    random.uniform(0, area_size[1])
                )
            
            # 围绕聚集中心生成位置
            r = random.uniform(5, 40)  # 聚集半径
            theta = random.uniform(0, 2 * np.pi)
            pos_x = cluster_center[0] + r * np.cos(theta)
            pos_y = cluster_center[1] + r * np.sin(theta)
            
            # 确保在区域内
            pos_x = max(0, min(area_size[0], pos_x))
            pos_y = max(0, min(area_size[1], pos_y))
        else:
            # 完全随机位置
            pos_x = random.uniform(0, area_size[0])
            pos_y = random.uniform(0, area_size[1])
        
        # 高度、直径和树冠 - 基于树种调整
        height_factor = 1.0
        if species == 'redwood':
            height_factor = 1.5
        elif species == 'pine' or species == 'fir':
            height_factor = 1.2
        elif species == 'birch' or species == 'ash':
            height_factor = 0.9
        
        height = random.uniform(height_range[0], height_range[1]) * height_factor
        
        # 树干直径与高度相关
        trunk_diameter = height * 0.05 + random.uniform(-0.1, 0.2)
        trunk_diameter = max(0.1, trunk_diameter)  # 确保不会是负值
        
        # 树冠大小与高度和树种相关
        canopy_factor = 1.0
        if species == 'oak' or species == 'maple':
            canopy_factor = 1.3
        elif species == 'pine' or species == 'fir':
            canopy_factor = 0.8
        
        canopy_size = random.uniform(canopy_range[0], canopy_range[1]) * canopy_factor
        
        # 构建树木数据
        tree = {
            'tree_id': i,
            'tree_species': species,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'pos_z': 0,  # 假设在同一平面
            'height': height,
            'trunk_diameter': trunk_diameter,
            'canopy_size': canopy_size
        }
        
        tree_data.append(tree)
    
    # 转换为DataFrame
    forest_df = pd.DataFrame(tree_data)
    
    # 森林参数
    forest_params = {
        'name': name,
        'area_size': area_size,
        'num_trees': num_trees,
        'tree_density': tree_density,
        'primary_species': primary_species
    }
    
    return forest_df, forest_params

def save_forest_data(forest_df, forest_params, output_dir="data"):
    """
    保存森林数据到指定目录
    
    参数:
        forest_df (pd.DataFrame): 树木数据表
        forest_params (dict): 森林参数
        output_dir (str): 输出目录
    """
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件名
    dataset_name = forest_params['name'].replace(" ", "_").lower()
    file_name = f"{dataset_name}_forest_data.xlsx"
    file_path = os.path.join(output_dir, file_name)
    
    # 保存到Excel
    forest_df.to_excel(file_path, index=False)
    
    print(f"森林数据已保存到: {file_path}")
    return file_path

def process_dataset_text(text):
    """处理数据集信息文本并生成数据"""
    print("== 处理数据集信息 ==")
    
    # 解析数据集信息
    datasets = parse_dataset_info(text)
    
    print(f"找到 {len(datasets)} 个数据集:")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset['name']}")
    
    # 为每个数据集生成森林数据
    for dataset in datasets:
        # 随机确定树木数量 - 根据森林类型
        name = dataset['name']
        if 'rainforest' in name or 'redwood' in name:
            num_trees = np.random.randint(60, 100)
        elif 'city_park' in name or 'downtown' in name or 'suburb' in name:
            num_trees = np.random.randint(30, 60)
        else:
            num_trees = np.random.randint(40, 80)
        
        generate_forest_data(dataset)

if __name__ == "__main__":
    # 示例数据集信息 (通常从用户输入获取)
    example_text = """
"meadow_forest": ["depth_pfm", "instance_segmentation", "rgb",
"semantic_segmentation"],
"birch_forest", coco_annotation, depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"broadleaf_forest", coco_annotation, depth_pfm, instance_segmentation,
landscape_info.txt, obj_info_final.xlsx, rgb, semantic_segmentation
"burned_forest", coco_annotation, depth_pfm, instance_segmentation, landscape_info.txt,
obj_info_final.xlsx, rgb, semantic_segmentation
"""
    
    # 从命令行参数获取文本文件路径
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            with open(file_path, 'r') as f:
                text = f.read()
            process_dataset_text(text)
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            print("将使用示例数据...")
            process_dataset_text(example_text)
    else:
        # 提示用户输入数据集信息
        try:
            print("请输入数据集信息 (按Ctrl+D结束输入):")
            user_input = []
            while True:
                try:
                    line = input()
                    user_input.append(line)
                except EOFError:
                    break
            
            if user_input:
                process_dataset_text("\n".join(user_input))
            else:
                print("未提供输入，将使用示例数据...")
                process_dataset_text(example_text)
        except Exception:
            print("处理输入时出错，将使用示例数据...")
            process_dataset_text(example_text) 