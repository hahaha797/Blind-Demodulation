# Blind-Demodulation。

## 数据集生成
- **DataSetGenerate**：生成数据集
  - pytorch格式
import torch
from torch.utils.data import DataLoader
import json

# 直接加载PyTorch Dataset
train_dataset = torch.load('{config.OUTPUT_DIR}/train_dataset.pt')
val_dataset = torch.load('{config.OUTPUT_DIR}/val_dataset.pt')
test_dataset = torch.load('{config.OUTPUT_DIR}/test_dataset.pt')

# 创建DataLoader（批量加载数据）
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载标签映射
with open('{config.OUTPUT_DIR}/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
idx_to_label = label_mapping['idx_to_label']

# 查看数据
for batch_data, batch_labels in train_loader:
    print(f"批量数据形状：{{batch_data.shape}}")  # (batch_size, 2, {config.SAMPLE_LENGTH})
    print(f"批量标签形状：{{batch_labels.shape}}")  # (batch_size,)
    print(f"前5个样本标签：{{[idx_to_label[str(l.item())] for l in batch_labels[:5]]}}")
    break

# 查看数据
for batch_data, batch_labels in train_loader:
    print(f"批量数据形状：{{batch_data.shape}}")  # (batch_size, 2, {config.SAMPLE_LENGTH})
    print(f"批量标签形状：{{batch_labels.shape}}")  # (batch_size,)
    print(f"前5个样本标签：{{[idx_to_label[str(l.item())] for l in batch_labels[:5]]}}")
    break
## 模型架构