import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA")

# 加载数据集
file_path = 'dataset_new/train_merge.pkl'
data = pd.read_pickle(file_path)

# Step 1: Convert the 'before_merge' column into TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=60)  # Limiting to top 60 features for simplicity
tfidf_matrix = vectorizer.fit_transform(data['before_merge']).toarray()
print("TF-IDF")

# Step 2: Define the GIN model
from torch.nn import Sequential, Linear, ReLU


class SimpleGIN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleGIN, self).__init__()
        self.conv1 = GINConv(Sequential(Linear(input_dim, 32), ReLU(), Linear(32, 32)))
        self.conv2 = GINConv(Sequential(Linear(32, output_dim), ReLU()))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Step 3: Create the GIN model and move it to the appropriate device
gin_model = SimpleGIN(input_dim=tfidf_matrix.shape[1], output_dim=10).to(
    device)  # output_dim=10 for vector representation
gin_model.eval()  # Set the model to evaluation mode
print("GIN model created")

# 定义批次大小
batch_size =4   # 可以根据GPU的内存调整

# 存储输出
gin_outputs = []
print("start")

# 分批处理数据
for i in range(0, len(tfidf_matrix), batch_size):
    print("batch", i)
    # 构建当前批次的输入数据
    x_batch = torch.tensor(tfidf_matrix[i:i + batch_size], dtype=torch.float).to(device)

    # 构建当前批次的边索引（这里我们使用局部邻域连接）
    batch_size_actual = x_batch.size(0)
    edge_index_batch = []
    for k in range(batch_size_actual):
        # 仅连接最近的10个节点，可以根据实际需求调整
        neighbors = list(range(max(0, k - 5), min(batch_size_actual, k + 6)))
        neighbors.remove(k)  # 移除自己节点的连接
        for j in neighbors:
            edge_index_batch.append([k, j])

    edge_index_batch = torch.tensor(edge_index_batch, dtype=torch.long).t().contiguous().to(device)

    with torch.no_grad():
        gin_output = gin_model(x_batch, edge_index_batch)

    # 将批次结果添加到输出列表
    gin_outputs.append(gin_output.cpu().numpy())
    print("batch:", i, "done")

# 将所有批次的结果合并
gin_outputs = np.vstack(gin_outputs)

# Step 4: Store the resulting vectors in a new column 'GIN'
data['GIN'] = gin_outputs.tolist()  # 转移到CPU上以便存储为Pandas的列

# 将DataFrame保存为Pickle文件
data.to_pickle('./GNN/train_merge_GIN.pkl')

print("DataFrame 已保存为 'new_test_GIN.pkl' 文件")
