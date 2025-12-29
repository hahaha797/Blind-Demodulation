# 调制信号深度学习数据集说明

## 数据集概述
- **名称**：调制信号数据集（Modulation Signal Dataset）
- **创建日期**：{pd.Timestamp.now().strftime('%Y-%m-%d')}
- **数据来源**：原始.bin和.wav格式调制信号文件
- **用途**：调制识别任务的深度学习模型训练、验证与测试

## 数据集结构
{OUTPUT_DIR}/
├── X_train.npy # 训练集特征（int16 格式）
├── X_val.npy # 验证集特征
├── X_test.npy # 测试集特征
├── y_train.npy # 训练集标签（整数编码）
├── y_val.npy # 验证集标签
├── y_test.npy # 测试集标签
├── label_mapping.json # 调制类型→整数标签映射表
├── metadata.csv # 样本元数据（文件名、采样率等）
└── README.md # 本说明文档

## 数据格式说明
### 1. 特征格式
- **数据类型**：numpy.ndarray（int16，取值范围：[-32768, 32767]）
- **形状**：(样本数, {feature_shape[0]}, {feature_shape[1]})
  - 维度1：样本数量（训练/验证/测试集分别独立）
  - 维度2：序列长度（每个样本含 {feature_shape[0]} 对IQ数据）
  - 维度3：信号通道（0=I路信号，1=Q路信号）

### 2. 标签格式
- **编码方式**：整数编码（LabelEncoder）
- **标签映射表**：
"""
    # 添加标签映射详情
    for mod, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = class_counts.get(mod, 0)
        readme_content += f"  - {idx} → {mod}（样本数：{count}）\n"

    readme_content += f"""
## 数据集统计信息
### 1. 样本分布
- **总样本数**：{total_samples}
- **训练集**：{len(X_train)} 样本（{len(X_train)/total_samples*100:.1f}%）
- **验证集**：{len(X_val)} 样本（{len(X_val)/total_samples*100:.1f}%）
- **测试集**：{len(X_test)} 样本（{len(X_test)/total_samples*100:.1f}%）

### 2. 类别分布
| 调制类型 | 样本数 | 占比 |
|----------|--------|------|
"""
    for mod, count in class_counts.items():
        readme_content += f"| {mod} | {count} | {count/total_samples*100:.1f}% |\n"

    readme_content += f"""
### 3. 采样率分布
- 包含采样率：{sample_rate_str}
- 说明：采样率为原始文件属性，未做统一归一化，详情见metadata.csv

## 数据预处理规则
### 1. 原始文件读取
- **.bin文件**：无数据头，按每 {SAMPLE_LENGTH} 对IQ分割为帧（样本），帧间不连续
- **.wav文件**：跳过1068字节头部，按 {SAMPLE_LENGTH} 对IQ分割样本，全文件连续
- 不足 {SAMPLE_LENGTH} 对IQ的数据片段直接丢弃，保证样本长度统一

### 2. 数据集划分
- 采用**分层抽样**：保证每个调制类型在训练/验证/测试集中的比例一致
- 划分比例：训练集70% → 验证集10% → 测试集20%
- 随机种子：{RANDOM_STATE}（可复现划分结果）

## 快速使用示例
# ```python```

import numpy as np
import json

# 加载数据
X_train = np.load('{OUTPUT_DIR}/X_train.npy')
y_train = np.load('{OUTPUT_DIR}/y_train.npy')
X_test = np.load('{OUTPUT_DIR}/X_test.npy')
y_test = np.load('{OUTPUT_DIR}/y_test.npy')

# 加载标签映射
with open('{OUTPUT_DIR}/label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
reverse_mapping = {{v: k for k, v in label_mapping.items()}}

# 查看数据信息
print(f"训练集特征形状：{{X_train.shape}}")  # (样本数, 131072, 2)
print(f"第一个样本标签：{{reverse_mapping[y_train[0]]}}")  # 调制类型名称