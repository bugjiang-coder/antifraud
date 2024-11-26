from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.data import Data
import os

# 设置数据集路径（需要替换为实际数据集路径）
# root = 'path/to/EllipticBitcoinDataset'
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/","EllipticBitcoinDataset")
# 加载数据集
dataset = EllipticBitcoinDataset(root=root)

# 检查数据集的一些统计信息
print(f"Number of graphs: {len(dataset)}")
print(f"Number of nodes: {dataset[0].num_nodes}")
print(f"Number of edges: {dataset[0].num_edges}")
print(f"Number of features per node: {dataset[0].num_node_features}")
print(f"Number of classes: {dataset.num_classes}")

# 查看特征的第一条记录，分析其是否包含时间特征
data = dataset[0]  # 取第一笔数据
print("First node features:", data.x[0])  # 打印第一个节点的特征

# 检查特征维度内容是否包含时间信息
# 通常，时间信息若存在可能以某种顺序或特定的特征维度出现
