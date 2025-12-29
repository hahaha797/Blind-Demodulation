# 调制识别BIDE-V1-1D模型训练脚本

该脚本基于PyTorch实现轻量化BIDE-V1-1D模型的训练流程，专用于调制识别任务，仅读取前序数据集构造脚本生成的npy格式数据，支持混合精度训练、梯度累积、学习率调度等优化策略，适配GPU训练场景。

## 目录
- [概述](#概述)
- [环境依赖](#环境依赖)
- [配置说明](#配置说明)
- [使用前提](#使用前提)
- [使用方法](#使用方法)
- [模型架构说明](#模型架构说明)
- [训练流程说明](#训练流程说明)
- [注意事项](#注意事项)
- [常见问题](#常见问题)

## 概述
本脚本核心功能：
- 仅加载npy格式的结构化数据集（避免重复解析原始IQ文件）
- 实现轻量化1D卷积模型（BIDE-V1-1D）的构建与训练
- 支持混合精度训练、梯度累积、学习率余弦退火调度
- 自动保存最优验证精度模型，并在训练结束后测试模型性能
- 实时监控训练损失、准确率、GPU显存使用情况

## 环境依赖
### 基础依赖
```bash
pip install torch torchvision torchaudio numpy tqdm
```
### 环境要求
- Python 3.7+
- PyTorch 1.8+（建议1.10+，低版本需兼容SiLU激活函数）
- CUDA 10.2+（建议GPU训练，CPU仅支持测试）
- 显存≥8GB（默认BATCH_SIZE=8适配8GB显存）

## 配置说明
脚本中`Config`类为核心配置区域，仅需修改该部分即可适配训练需求：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATASET_OUTPUT_DIR` | 字符串 | `./modulation_dataset` | 数据集构造脚本的输出目录（需包含npy文件） |
| `BATCH_SIZE` | 整数 | 8 | 批次大小（8GB显存建议8，16GB可设16/32） |
| `EPOCHS` | 整数 | 5 | 训练轮数（测试用，正式训练建议设50+） |
| `LR` | 浮点数 | 1e-4 | 初始学习率（AdamW优化器） |
| `WEIGHT_DECAY` | 浮点数 | 1e-5 | 权重衰减（防止过拟合） |
| `ACCUMULATION_STEPS` | 整数 | 4 | 梯度累积步数（等效增大批次：8×4=32） |
| `NUM_CLASSES` | 整数 | 20 | 调制类型数量（需与数据集构造脚本的标签数一致） |

## 使用前提
1. 已运行`dataset_constructor.py`生成完整的npy数据集，确保`DATASET_OUTPUT_DIR`目录下包含以下文件：
   - `train_data.npy` / `train_labels.npy`
   - `val_data.npy` / `val_labels.npy`
   - `test_data.npy` / `test_labels.npy`
2. 确认GPU可用（脚本默认使用CUDA，无GPU需修改设备配置）

## 使用方法

### 1. 配置参数
修改脚本中`Config`类的参数，核心确认：
- `DATASET_OUTPUT_DIR`：指向数据集构造脚本的输出目录
- `BATCH_SIZE`：根据GPU显存调整（8GB设8，16GB设16）
- `EPOCHS`：测试阶段设5，正式训练设50+
- `NUM_CLASSES`：与实际调制类型数量一致（需匹配label_mapping.json中的类别数）

### 2. 运行训练脚本
```bash
python trainer.py  # 替换为你的脚本实际文件名
```
运行过程中会输出：
- 数据集加载验证信息
- 每轮训练的进度条（含损失、准确率、显存使用）
- 验证阶段结果
- 最优模型保存提示
- 最终测试集准确率

### 3. 训练结果
训练完成后，在`DATASET_OUTPUT_DIR`目录下生成：
- `BIDE-V1_1d_best.pth`：最优验证精度模型权重（包含模型参数、优化器状态、最优准确率）

## 模型架构说明
### BIDE-V1_1D_Classifier 结构
| 模块 | 说明 |
|------|------|
| Backbone（特征提取） | 5层1D卷积堆叠：<br>- 输入：2通道（I/Q）×4096长度<br>- 卷积核：6/3，步长2，逐层下采样<br>- 激活：SiLU/Swish（低版本PyTorch兼容）<br>- 归一化：BatchNorm1d |
| Class Head（分类头） | - AdaptiveAvgPool1d：全局平均池化（降维至128维）<br>- Flatten：展平特征<br>- 全连接层：128→64→NUM_CLASSES<br>- 正则：LayerNorm + Dropout(0.1) |
| 权重初始化 | 卷积/全连接层使用Kaiming正态初始化 |

### 输入输出
- 输入：`[B, 2, 4096]`（B=批次大小，2=I/Q通道，4096=IQ序列长度）
- 输出：`[B, NUM_CLASSES]`（各类别的预测概率对数）

## 训练流程说明
1. **数据校验**：检查必需的npy文件是否存在，缺失则终止并提示
2. **设备初始化**：优先使用CUDA，启用cudnn加速，适配Windows系统路径策略
3. **数据集加载**：通过`NpyDataset`类加载npy数据，转换为PyTorch张量
4. **DataLoader构建**：训练集打乱，验证/测试集不打乱，批次大小适配
5. **模型初始化**：构建BIDE-V1-1D模型，加载至GPU，初始化损失函数/优化器/调度器
6. **训练循环**：
   - 训练阶段：混合精度训练 + 梯度累积，每10批次清理GPU显存
   - 验证阶段：每轮训练后评估验证集，计算损失和准确率
   - 模型保存：保存验证准确率最优的模型权重
7. **测试阶段**：加载最优模型，评估测试集准确率并输出最终结果

## 注意事项
1. **显存优化**：
   - 若显存不足，减小`BATCH_SIZE`（如设4）或增大`ACCUMULATION_STEPS`
   - 每10批次自动清理GPU显存，避免内存泄漏
2. **混合精度训练**：使用`GradScaler`加速训练，降低显存占用
3. **梯度累积**：等效增大批次大小，提升训练稳定性（适合小显存场景）
4. **学习率调度**：采用余弦退火调度，随训练轮数逐步降低学习率
5. **设备适配**：
   - 无GPU时，需将代码中`'cuda'`改为`'cpu'`（包括模型初始化、数据传输）
   - Windows系统已适配CUDA可见设备和多进程策略
6. **过拟合预防**：使用权重衰减、Dropout、LayerNorm等正则化手段

## 常见问题
### Q1: 提示找不到npy文件？
A1: 检查`DATASET_OUTPUT_DIR`配置是否正确，确保已运行数据集构造脚本生成完整的npy文件。

### Q2: CUDA out of memory（显存不足）？
A2: 
- 减小`BATCH_SIZE`（如从8改为4）
- 增大`ACCUMULATION_STEPS`（保持等效批次）
- 关闭其他占用GPU的程序
- 降低`SAMPLE_LENGTH`（需重新生成数据集）

### Q3: 训练准确率低/不收敛？
A3:
- 增大训练轮数（`EPOCHS`设50+）
- 调整学习率（如将LR改为5e-4）
- 确认`NUM_CLASSES`与实际类别数一致
- 检查数据集是否存在标签错误或样本分布不均

### Q4: 低版本PyTorch报错SiLU不存在？
A4: 脚本已内置Swish类兼容低版本PyTorch，无需额外修改，自动切换激活函数。

### Q5: 模型保存后加载失败？
A5: 确保加载时的模型结构与训练时一致（尤其是`NUM_CLASSES`），加载代码示例：
```python
model = BIDE-V1_1D_Classifier(num_classes=20)
checkpoint = torch.load("BIDE-V1_1d_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```