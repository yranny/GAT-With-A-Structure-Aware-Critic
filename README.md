GAT-With-A-Structure-Aware-Critic/
├── .venv/                    ← 虚拟环境
├── requirements.txt
├── src/
│   ├── model_vqa_gat.py      ← GAT 模型
│   ├── symbolic_layout.py    ← layout parser
│   ├── graph_builder.py
│   ├── forest_simulation.py  ← 森林模拟脚本
│   ├── forest_gat_analysis.py ← 森林GAT分析脚本
│   ├── parse_dataset_info.py ← 数据集信息解析脚本
│   ├── real_forest_visualization.py ← 真实森林可视化脚本
│   └── test_model.py         ← 测试代码（你可以运行）
├── data/                     ← 数据目录
│   └── forest_data.xlsx      ← 模拟森林数据
├── datasets.txt              ← 数据集信息文件
└── results/                  ← 输出目录
    ├── forest_visualization.png       ← 基本森林可视化
    ├── forest_gat_analysis.png        ← 森林GAT分析可视化
    └── real_forest/                   ← 真实森林可视化结果
        ├── rainforest_gat.png         ← 雨林GAT分析
        ├── city_park_gat.png          ← 城市公园GAT分析
        └── ...                        ← 其他森林类型分析

## 功能概述

### 图构建 (Graph Construction)
基于森林数据构建图结构，树木作为节点，树木间的关系作为边。

### GAT模型 (GAT Model)
使用图注意力网络分析树木之间的关系权重，发现重要连接。

### 结构感知评论器 (Structure-Aware Critic)
分析树木的空间分布和种群结构，评估生态系统健康度。

### 训练和损失函数 (Training & Loss)
用于训练模型的损失函数设计。

### 评估与可视化 (Evaluation & Visualization)
提供多种可视化方式展示模型结果：
- 基本森林可视化：展示树木分布和物种类型
- GAT注意力可视化：展示树木间重要关系
- 多种森林类型分析：包括雨林、红木林、城市公园等

## 使用方法

### 森林模拟
```bash
python src/forest_simulation.py
```

### 森林GAT分析
```bash
python src/forest_gat_analysis.py
```

### 真实森林数据分析
```bash
python src/real_forest_visualization.py datasets.txt
```
或者直接运行:
```bash
python src/real_forest_visualization.py
```
程序会提示输入数据集信息或使用默认数据集。

