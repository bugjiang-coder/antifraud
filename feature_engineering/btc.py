import pandas as pd

# 读取 CSV 文件
file_path = 'data/EllipticBitcoinDataset/raw/elliptic_txs_features.csv'  # 替换为实际文件路径
df = pd.read_csv(file_path, header=None)

# 查看前几行数据
print("数据预览：")
print(df.head())

# 查看数据集的基本信息
print("\n数据集信息：")
print(df.info())

# 查看缺失值情况
print("\n缺失值检查：")
print(df.isnull().sum())

# 查看数据的统计信息
print("\n统计信息：")
print(df.describe())

# 假设时间信息在特定列，如第一个特征列。检查其是否与时间相关
# 假设列索引0为交易ID，列索引1可能为时间
print("\n前10个交易的时间特征：")
print(df[0].head(10))  # 观察第1列的数值分布是否和时间相关

# 检查数据集的整体特征数量
num_features = df.shape[0] - 2  # 排除第一个列（假设为ID）和最后一列（可能为标签）
print(f"\n总特征数量（不包括ID和标签）：{num_features}")

# 检查标签分布（假设最后一列为标签）
print("\n标签分布：")
print(df[df.columns[-1]].value_counts())
