# YOLO12-1D 调制信号识别模型架构说明文档
## 文档概述
本文档针对调制信号识别任务中使用的 **YOLO12-1D 模型** 进行详细架构解析，该模型是 YOLO12 目标检测架构的 1 维卷积适配版本，专为 IQ 双通道调制信号（时序特征）设计，兼具轻量化、高算力效率和高精度特性，适配 Float16 数据精度与 48G 显存环境，是调制信号分类任务的核心网络结构。

### 核心定位
- 任务场景：多类调制信号（如 2FSK、16QAM、QPSK 等）的分类识别；
- 输入特征：IQ 双通道时序信号（形状 `[B, 2, 4096]`，Float16/Float32）；
- 设计目标：在 48G 显存环境下高效训练，平衡特征提取能力与显存/计算资源占用；
- 核心优势：1D 卷积适配时序特征、多阶段下采样提取层级特征、全局上下文聚合提升分类精度。

## 一、核心设计理念
### 1. 1D 卷积适配 IQ 信号特性
IQ 调制信号是**双通道时序数据**（I 路/同相路、Q 路/正交路），传统 2D 卷积（面向图像）无法有效捕捉时序特征，因此模型全部采用 1D 卷积层，沿时间维度提取特征。

### 2. 轻量化与层级特征提取
通过 6 阶段 1D 卷积下采样，逐步将 4096 长度的时序信号压缩为 64 长度的高维特征，既保证特征抽象能力，又控制参数量（整体参数量约百万级），适配大规模数据集训练。

### 3. 双通道特征融合
输入层直接接收 `[B, 2, 4096]` 形状的 IQ 双通道数据，首层卷积将通道数从 2 扩展至 32，实现双通道特征的早期融合，充分利用 IQ 信号的相位/幅度关联信息。

### 4. 适配 Float16 数据精度
模型权重初始化、归一化层设计均兼容 Float16 数据输入，配合混合精度训练，可将显存占用降低 50%，适配 48G 显存下大批次训练。

## 二、模型架构总览
### 1. 输入输出定义
| 维度 | 说明 | 形状（Batch Size=B） | 数据类型 |
|------|------|----------------------|----------|
| 输入 | IQ 双通道时序信号 | `[B, 2, 4096]` | Float16/Float32 |
| 输出 | 调制类型分类概率 | `[B, NUM_CLASSES]` | Float32 |

### 2. 架构整体流程
```
输入 [B,2,4096] → Backbone（6阶段1D卷积）→ 全局平均池化 → 分类头 → 输出 [B,NUM_CLASSES]
```

## 三、模块详细解析
### 1. 基础组件定义
#### （1）激活函数适配
模型优先使用 PyTorch 原生 `nn.SiLU()`（Swish 激活），低版本 PyTorch 兼容自定义 Swish 实现：
```python
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)
```
SiLU 激活相比 ReLU 更平滑，能更好捕捉调制信号的非线性特征，且梯度消失风险更低。

#### （2）权重初始化
- 卷积层：Kaiming 正态初始化（`mode='fan_out'`，`nonlinearity='relu'`），适配 SiLU/Swish 激活的梯度分布；
- 批归一化层：权重初始化为 1，偏置初始化为 0，保证初始分布稳定。

### 2. Backbone（特征提取网络）
Backbone 由 6 个卷积阶段组成，每阶段包含「1D 卷积 + 批归一化 + 激活函数」，逐步下采样并提升通道数，提取从浅层到深层的时序特征。

| 阶段 | 卷积配置 | 输出形状（输入 `[B,2,4096]`） | 核心作用 |
|------|----------|------------------------------|----------|
| Stage 1 | `Conv1d(2, 32, 7, stride=2, padding=3, bias=False)` + BN + SiLU | `[B, 32, 2048]` | 双通道特征融合，首次下采样（步长2），提取基础时序特征 |
| Stage 2 | `Conv1d(32, 64, 5, stride=2, padding=2, bias=False)` + BN + SiLU + Dropout1d(0.1) | `[B, 64, 1024]` | 通道数翻倍，二次下采样，Dropout 防止过拟合 |
| Stage 3 | `Conv1d(64, 128, 3, stride=2, padding=1, bias=False)` + BN + SiLU | `[B, 128, 512]` | 通道数翻倍，三次下采样，提取中层特征 |
| Stage 4 | `Conv1d(128, 256, 3, stride=2, padding=1, bias=False)` + BN + SiLU | `[B, 256, 256]` | 通道数翻倍，四次下采样，提取高阶时序特征 |
| Stage 5 | `Conv1d(256, 512, 3, stride=2, padding=1, bias=False)` + BN + SiLU | `[B, 512, 128]` | 通道数翻倍，五次下采样，抽象调制信号核心特征 |
| Stage 6 | `Conv1d(512, 512, 3, stride=2, padding=1, bias=False)` + BN + SiLU | `[B, 512, 64]` | 保持通道数，六次下采样，全局上下文特征提取 |

#### 关键设计细节
- 卷积核尺寸：浅层用 7/5 核（捕捉长时序特征），深层用 3 核（捕捉局部精细特征）；
- 步长策略：每阶段步长=2，6 阶段后时序长度从 4096 压缩至 64（下采样 64 倍）；
- 偏置设置：卷积层禁用偏置（`bias=False`），由批归一化层替代，减少参数并提升稳定性；
- Dropout 仅在 Stage 2 引入（`Dropout1d(0.1)`），平衡过拟合与特征保留。

### 3. 分类头（Classifier）
分类头负责将 Backbone 输出的高维特征映射为调制类型分类结果，包含「全局平均池化 + 全连接层 + 归一化 + Dropout」：

| 层类型 | 配置 | 输入形状 | 输出形状 | 作用 |
|--------|------|----------|----------|------|
| 全局平均池化 | `AdaptiveAvgPool1d(1)` | `[B, 512, 64]` | `[B, 512, 1]` | 聚合全局时序特征，消除长度维度，输出固定维度特征 |
| 展平 | `nn.Flatten()` | `[B, 512, 1]` | `[B, 512]` | 转为一维特征向量，适配全连接层 |
| 全连接层1 | `nn.Linear(512, 256)` | `[B, 512]` | `[B, 256]` | 特征维度压缩，降低计算量 |
| 层归一化 | `nn.LayerNorm(256)` | `[B, 256]` | `[B, 256]` | 适配小批次训练，提升稳定性（相比 BN 更适合分类头） |
| 激活函数 | SiLU/Swish | `[B, 256]` | `[B, 256]` | 引入非线性，增强特征表达 |
| Dropout | `nn.Dropout(0.3)` | `[B, 256]` | `[B, 256]` | 随机失活，防止过拟合 |
| 全连接层2 | `nn.Linear(256, NUM_CLASSES)` | `[B, 256]` | `[B, NUM_CLASSES]` | 输出分类概率（未激活，配合 CrossEntropyLoss） |

#### 关键设计细节
- 全局平均池化（AdaptiveAvgPool1d）：无需手动指定池化核尺寸，自适应压缩时序维度，保证输入长度变化时模型仍可用；
- 层归一化（LayerNorm）：分类头使用 LayerNorm 而非 BatchNorm，避免小批次下归一化统计不稳定；
- Dropout 概率 0.3：在分类头引入适度正则化，平衡调制信号分类的精度与泛化能力。

## 四、模型参数与显存估算
### 1. 参数量计算（以 NUM_CLASSES=20 为例）
| 模块 | 参数量（约） | 占比 |
|------|--------------|------|
| Backbone | 1.8M | 85.7% |
| 分类头 | 0.3M | 14.3% |
| 总计 | 2.1M | 100% |

### 2. 显存占用估算（Float16 输入，Batch Size=64）
| 部分 | 显存占用（约） | 说明 |
|------|----------------|------|
| 输入数据 | 64×2×4096×2Byte = 1MB | Float16 单元素 2Byte |
| Backbone 特征 | 约 1.2GB | 各阶段特征图显存总和 |
| 分类头特征 | 约 0.1GB | 全连接层中间特征 |
| 梯度/优化器状态 | 约 2.4GB | 混合精度训练下梯度缩放占用 |
| 总计（单批次） | 约 3.7GB | 48G 显存可支持 Batch Size=64~128 |

> 注：梯度累积（ACCUMULATION_STEPS=4）不会增加单批次显存占用，仅等效增大有效批次，是 48G 显存下提升训练稳定性的核心策略。

## 五、适配优化点（针对 Float16/48G 显存）
### 1. Float16 数据兼容
- 权重初始化：Kaiming 初始化适配 Float16 精度，避免梯度下溢；
- 归一化层：BN/LN 层的均值/方差计算使用 Float32 中间值，防止 Float16 精度丢失；
- 混合精度训练：配合 `torch.cuda.amp.GradScaler`，前向传播用 Float16，反向传播用 Float32 梯度，平衡精度与显存。

### 2. 显存优化
- 卷积层禁用偏置：减少约 5% 的参数显存占用；
- 分阶段下采样：逐步压缩特征图尺寸，避免大尺寸特征图长期占用显存；
- 梯度累积：Batch Size=64 + ACCUMULATION_STEPS=4 等效 Batch Size=256，无需增大单批次显存。

### 3. 训练效率优化
- 非阻塞数据传输：`to(device, non_blocking=True)` 加速 CPU→GPU 数据传输；
- OneCycleLR 学习率策略：适配模型小参数量特性，快速收敛且避免过拟合；
- 标签平滑（CrossEntropyLoss(label_smoothing=0.1)）：提升模型泛化能力，适配调制信号的类内差异。

## 六、模型使用说明
### 1. 实例化方式
```python
from model import YOLO12_1D_Modulation

# 初始化模型（NUM_CLASSES 从 label_mapping.json 读取）
num_classes = 20  # 调制类型数量
model = YOLO12_1D_Modulation(num_classes=num_classes).to("cuda")
```

### 2. 输入输出要求
- 输入：必须为 `[B, 2, 4096]` 形状的张量，Float16/Float32 均可（模型自动兼容）；
- 输出：`[B, NUM_CLASSES]` 形状的未激活张量，需配合 `nn.CrossEntropyLoss` 计算损失。

### 3. 训练适配建议
- 48G 显存：Batch Size=64，ACCUMULATION_STEPS=4，学习率 3e-4；
- 24G 显存：Batch Size=32，ACCUMULATION_STEPS=8，学习率 1.5e-4；
- 混合精度训练：必须使用 `torch.cuda.amp.autocast()` 和 `GradScaler`，否则 Float16 输入可能导致精度丢失。

## 七、扩展优化建议
### 1. 特征增强
- 添加 1D 注意力机制（如 SE Block、CBAM）：在 Stage 5/6 后引入，增强关键时序特征的权重；
- 多尺度特征融合：融合 Stage 4/5/6 的特征，捕捉不同粒度的调制信号特征；
- 残差连接：在 Backbone 各阶段添加 1D 残差块，解决深层网络梯度消失问题。

### 2. 轻量化优化
- 通道剪枝：将 Backbone 通道数按比例缩减（如 32→16、64→32），适配小显存环境；
- 深度可分离卷积：替换部分普通 1D 卷积为深度可分离卷积，参数量减少 80% 以上；
- 量化训练：导出 INT8 量化模型，部署时提升推理速度。

### 3. 任务适配
- 多标签分类：修改分类头输出激活为 Sigmoid，适配多调制类型共存场景；
- 信噪比感知：添加信噪比分支，联合训练调制类型与信噪比分类；
- 端到端增强：在输入层添加噪声抑制、相位校准等前置模块，提升低信噪比下的精度。

## 八、总结
YOLO12-1D 模型是针对 IQ 调制信号分类任务的轻量化、高效架构，通过 1D 卷积适配时序特征、多阶段下采样提取层级特征、全局上下文聚合提升分类精度，同时深度适配 Float16 数据精度与 48G 显存环境。模型参数量仅 2.1M，训练效率高且泛化能力强，是大规模调制信号识别任务的理想选择，可通过注意力机制、残差连接等扩展进一步提升性能。

# 调制信号识别模型训练代码说明文档
## 文档概述
本文档针对**基于YOLO12-1D的调制信号识别模型训练代码**进行详细说明，该代码专为 Float16 格式的 IQ 信号数据集设计，适配 48G 大内存/GPU 环境，核心优化了内存占用、显存利用效率，支持大规模调制信号样本的高效训练。

### 核心适配场景
- 数据集格式：Float16 格式的 `.npy` 文件（IQ 信号，形状 [N, 2, 4096]）；
- 硬件环境：48G 显存 GPU + 大内存服务器/工作站；
- 业务场景：多类调制信号（如 2FSK、16QAM、QPSK 等）的分类识别。

### 代码版本说明
本文档整合了两段代码的核心逻辑，重点突出**48G 内存/GPU 适配优化**和 **Float16 数据集兼容**，修复了基础版代码的内存占用、Windows 系统兼容等问题。

## 一、核心配置参数详解
代码中 `Config` 类是训练的核心配置入口，所有参数均针对 48G 内存/GPU 环境优化，参数说明如下：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATASET_OUTPUT_DIR` | 字符串 | `./modulation_dataset_50overlap` | 数据集根目录，需包含 `train_data.npy`、`val_data.npy`、`test_data.npy` 及标签文件、`label_mapping.json` |
| `LOG_DIR` | 字符串 | `./train_logs` | 训练日志保存目录，自动生成 `train_log.txt` |
| `BATCH_SIZE` | 整数 | 64 | 批次大小（48G 显存推荐 64-128），Float16 数据显存占用减半，可比 Float32 场景增大 1 倍 |
| `EPOCHS` | 整数 | 50 | 训练总轮数，正式训练建议 50-100 轮 |
| `LR` | 浮点数 | 3e-4 | AdamW 优化器初始学习率 |
| `WEIGHT_DECAY` | 浮点数 | 1e-4 | 权重衰减系数，防止模型过拟合 |
| `ACCUMULATION_STEPS` | 整数 | 4 | 梯度累积步数，等效批次 = `BATCH_SIZE * ACCUMULATION_STEPS`（64*4=256），显存不足时可增大该值、减小 `BATCH_SIZE` |
| `WARMUP_EPOCHS` | 整数 | 3 | 学习率预热轮数（OneCycleLR 策略） |
| `NUM_CLASSES` | 整数 | 0 | 自动从 `label_mapping.json` 读取，无需手动修改 |
| `SAMPLE_LENGTH` | 整数 | 4096 | IQ 信号样本长度，需与数据集一致 |
| `SAVE_INTERVAL` | 整数 | 5 | 模型检查点保存间隔（每 N 轮保存一次） |
| `MAX_GPU_MEM_RATIO` | 浮点数 | 0.90 | GPU 显存最大占用比例，防止显存溢出 |

## 二、代码模块功能说明
### 1. 工具函数模块
#### （1）`log_info(msg, save_to_file=True)`
- 功能：打印训练日志并保存到文件，日志包含时间戳、训练指标等；
- 参数：`msg` 为日志内容，`save_to_file` 控制是否保存到 `train_log.txt`；
- 输出：控制台打印 + `LOG_DIR/train_log.txt` 文件写入。

#### （2）`monitor_resources()`
- 功能：实时监控 GPU 显存（已分配/已保留）、CPU 内存使用率；
- 适配：仅在 CUDA 可用时显示 GPU 信息，否则显示 CPU；
- 用途：训练过程中监控资源占用，避免显存/内存溢出。

### 2. 数据集类（`ModulationDataset`）
核心适配 Float16 数据集和 48G 内存的关键模块，解决大规模 `.npy` 文件加载的内存占用问题。

#### 核心特性
| 特性 | 实现方式 | 优势 |
|------|----------|------|
| 内存映射加载 | `np.load(..., mmap_mode='r')` | 数据保留在硬盘，随用随取，避免几十G数据集一次性加载到内存 |
| Float16 兼容 | `astype(np.float32)` 转换 | 适配模型 Float32 输入要求，同时保留数据集 Float16 存储的体积优势 |
| 形状校验 | 检查数据形状为 [N, 2, 4096] | 提前发现数据集格式错误，避免训练中断 |
| 容错处理 | 异常样本返回全零张量 + 标签0 | 单个样本加载失败不影响整体训练 |

#### 关键方法
- `__init__`：初始化时加载数据集（内存映射），校验文件存在性和形状；
- `__len__`：返回样本总数；
- `__getitem__`：读取单个样本，完成 Float16→Float32 转换、张量封装。

### 3. 模型定义（`YOLO12_1D_Modulation`）
轻量化 1D 卷积模型，专为 IQ 双通道信号设计，适配 4096 长度的调制信号特征提取。

#### 模型结构
| 模块 | 结构 | 输出维度（输入 [B,2,4096]） |
|------|------|----------------------------|
| Backbone（特征提取） | 6层 1D 卷积 + BN + 激活函数 | [B,512,64] |
| 全局平均池化 | `AdaptiveAvgPool1d(1)` | [B,512,1] |
| 分类头 | Flatten + 全连接 + Dropout | [B, NUM_CLASSES] |

#### 激活函数适配
- 自动选择 `nn.SiLU()`（高版本 PyTorch）或自定义 `Swish`（低版本兼容）；
- 权重初始化：Conv1d 用 Kaiming 初始化，BN 层权重置1、偏置置0。

### 4. 训练主流程（`train_model`）
整合模型训练、验证、测试、模型保存的全流程，针对 48G 内存/GPU 优化训练效率。

#### 核心步骤
1. **设备初始化**：自动选择 CUDA/CPU，优先使用 GPU；
2. **类别数自动读取**：从 `label_mapping.json` 读取调制类别数，兼容两种 JSON 格式；
3. **数据加载器配置**：
   - Windows 下 `num_workers=0`（避免多进程与内存映射冲突）；
   - Linux 下 `num_workers=4/8`（提升加载速度）；
   - 验证集/测试集批次大小为训练集的 2 倍（提升评估效率）；
4. **优化器与学习率策略**：
   - 优化器：AdamW（带权重衰减，防止过拟合）；
   - 学习率策略：OneCycleLR（含预热，提升收敛速度）；
5. **混合精度训练**：`GradScaler` + `autocast()`，降低显存占用，提升训练速度；
6. **梯度累积**：按 `ACCUMULATION_STEPS` 累积梯度后更新参数，等效增大批次；
7. **模型保存**：
   - 保存验证集最优模型（`best_model.pth`）；
   - 按 `SAVE_INTERVAL` 保存轮次检查点；
8. **测试评估**：训练结束后加载最优模型，评估测试集准确率。

#### 评估函数（`evaluate`）
- 功能：模型验证/测试，计算损失和准确率；
- 优化：`torch.no_grad()` + 混合精度，减少显存占用。

## 三、关键优化点（适配48G内存/GPU）
### 1. 内存占用优化
- 内存映射加载数据集：避免几十G Float16 数据集占用物理内存；
- Windows 系统兼容：`num_workers=0` 解决内存映射文件句柄冲突问题；
- 分批次加载：仅在训练时读取当前批次数据，不缓存全量数据。

### 2. 显存利用优化
- Float16 数据集适配：存储为 Float16，加载时转为 Float32，显存占用减半；
- 混合精度训练：`GradScaler` + `autocast()`，进一步降低显存占用；
- 梯度累积：等效增大批次的同时，不增加单批次显存占用；
- 合理批次大小：48G 显存设置 `BATCH_SIZE=64`，平衡训练速度和显存占用。

### 3. 训练效率优化
- OneCycleLR 学习率策略：含预热和余弦衰减，加速模型收敛；
- 标签平滑：CrossEntropyLoss 带 `label_smoothing=0.1`，提升模型泛化能力；
- 非阻塞数据传输：`non_blocking=True` 加速数据从 CPU 到 GPU 的传输。

## 四、运行步骤
### 1. 环境准备
安装依赖包（建议使用 Conda 环境）：
```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 其他依赖
pip install numpy tqdm psutil
```

### 2. 数据集准备
- 数据集目录结构：
  ```
  modulation_dataset_50overlap/
  ├── train_data.npy       # 训练集数据（Float16, [N,2,4096]）
  ├── train_labels.npy     # 训练集标签（整数）
  ├── val_data.npy         # 验证集数据
  ├── val_labels.npy       # 验证集标签
  ├── test_data.npy        # 测试集数据
  ├── test_labels.npy      # 测试集标签
  └── label_mapping.json   # 调制类型-标签映射（字典格式）
  ```
- 确认 `label_mapping.json` 格式：
  ```json
  {"2FSK":0, "16QAM":1, "QPSK":2, ...}
  # 或兼容格式
  {"label_to_idx":{"2FSK":0}, "idx_to_label":{"0":"2FSK"}, "num_classes":3}
  ```

### 3. 启动训练
```python
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Windows 必加，防止多进程报错
    train_model()
```
直接运行代码，训练过程会在控制台打印日志，并保存到 `train_logs/train_log.txt`。

### 4. 结果查看
- 日志文件：`train_logs/train_log.txt`（包含每轮训练/验证损失、准确率、资源占用）；
- 模型文件：
  - 最优模型：`modulation_dataset_50overlap/best_model.pth`；
  - 检查点模型：`modulation_dataset_50overlap/epoch_5.pth`（每5轮保存）；
- 关键指标：训练集/验证集/测试集的损失值、Top-1 准确率。

## 五、常见问题与解决方案
### 1. Windows 下报错 `BrokenPipeError`
- 原因：`num_workers>0` 与内存映射（`mmap_mode='r'`）冲突；
- 解决方案：保持 `num_workers=0`（代码已默认适配）。

### 2. 显存溢出（`CUDA out of memory`）
- 解决方案：
  1. 减小 `BATCH_SIZE`（如从 64 改为 32）；
  2. 增大 `ACCUMULATION_STEPS`（如从 4 改为 8）；
  3. 降低 `MAX_GPU_MEM_RATIO`（如从 0.9 改为 0.8）；
  4. 关闭其他占用 GPU 的程序。

### 3. 数据集加载失败（`FileNotFoundError`）
- 原因：数据集路径错误或文件缺失；
- 解决方案：
  1. 检查 `Config.DATASET_OUTPUT_DIR` 路径是否正确；
  2. 确认目录下包含 `train_data.npy`、`train_labels.npy` 等文件；
  3. 检查文件扩展名（需为 `.npy`，而非 `.npz` 或其他）。

### 4. 类别数读取错误
- 原因：`label_mapping.json` 格式错误；
- 解决方案：
  1. 确保 JSON 文件为标准字典格式；
  2. 若文件缺失，代码会默认使用 20 类，需手动确认调制类别数并修改 `config.NUM_CLASSES`。

### 5. 训练准确率为 0
- 原因：
  1. 标签与数据不匹配；
  2. 学习率过高/过低；
  3. 数据集样本长度不为 4096；
- 解决方案：
  1. 校验数据集形状（需为 [N,2,4096]）；
  2. 调整学习率（如 3e-4 改为 1e-4）；
  3. 检查标签是否正确映射到调制类型。

## 六、扩展建议
1. **数据增强**：在 `ModulationDataset.__getitem__` 中添加高斯噪声、相位偏移、幅度缩放等增强，提升模型泛化能力；
2. **多GPU训练**：适配 `torch.nn.DataParallel` 或 `torch.distributed`，进一步提升训练速度；
3. **模型轻量化**：减少卷积层通道数（如 512 改为 256），适配小显存环境；
4. **学习率调优**：根据训练曲线调整 `OneCycleLR` 的 `pct_start` 参数，优化收敛速度。

## 七、总结
本代码针对 Float16 格式的调制信号数据集和 48G 内存/GPU 环境做了深度优化，核心解决了大规模数据集加载的内存占用问题、显存利用效率问题，同时保证了训练稳定性和模型性能。通过配置参数的灵活调整，可适配不同显存/内存规模的硬件环境，满足调制信号识别的训练需求。

graph TD
    %% 定义样式
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef embed fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef concat fill:#ffe0b2,stroke:#f57c00,stroke-width:2px;
    classDef block fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef se fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5;
    classDef pool fill:#e0f7fa,stroke:#006064,stroke-width:2px;
    classDef dense fill:#fce4ec,stroke:#880e4f,stroke-width:2px;

    %% 1. 输入层
    In[Input I/Q Signal<br/>(Batch, 2, 4096)]:::input --> Split

    %% 2. 多域特征嵌入 (Multi-Domain Embedding)
    subgraph Embedding_Layer [Multi-Domain Embedding Layer]
        direction TB
        Split((Split)) --> Branch1[Time Domain<br/>Pass Through]
        Split --> Branch2[Freq Domain<br/>FFT -> Abs -> Log1p]
        Split --> Branch3[Time-Freq Domain<br/>Haar Wavelet Conv]
        
        Branch1 -- 2 Channels --> Concat
        Branch2 -- 1 Channel --> Concat
        Branch3 -- 4 Channels --> Concat
        
        Concat[Concatenate<br/>(Batch, 7, 4096)]:::concat
    end

    %% 3. 特征提取骨干 (Backbone)
    subgraph Backbone [Feature Extractor with SE-Blocks]
        direction TB
        Concat --> S1
        
        subgraph Stage1 [Stage 1]
            S1[Conv1d (k=7, s=2)<br/>BN + SiLU]:::block
            SE1(SE-Block r=4):::se
            S1 --> SE1
        end
        
        SE1 --> S2_In
        
        subgraph Stage2 [Stage 2]
            S2_In[Conv1d (k=5, s=2)<br/>BN + SiLU]:::block
            SE2(SE-Block r=8):::se
            Drop1[Dropout 0.1]
            S2_In --> SE2 --> Drop1
        end

        Drop1 --> S3_In
        
        subgraph Stage3 [Stage 3]
            S3_In[Conv1d (k=3, s=2)<br/>BN + SiLU]:::block
            SE3(SE-Block r=16):::se
            S3_In --> SE3
        end

        SE3 --> S4_In
        
        subgraph Stage4 [Stage 4]
            S4_In[Conv1d (k=3, s=2)<br/>BN + SiLU]:::block
            SE4(SE-Block r=16):::se
            S4_In --> SE4
        end

        SE4 --> S5_In
        
        subgraph Stage5 [Stage 5]
            S5_In[Conv1d (k=3, s=2)<br/>BN + SiLU]:::block
            SE5(SE-Block r=16):::se
            S5_In --> SE5
        end

        SE5 --> S6_In

        subgraph Stage6 [Stage 6]
            S6_In[Conv1d (k=3, s=2)<br/>BN + SiLU]:::block
            SE6(SE-Block r=16):::se
            S6_In --> SE6
        end
    end

    %% 4. 分类头 (Classifier Head)
    SE6 -- (Batch, 512, 64) --> Pool[Adaptive AvgPool1d<br/>(Global Pooling)]:::pool
    Pool -- (Batch, 512, 1) --> Flat[Flatten]:::dense
    Flat -- (Batch, 512) --> FC1[Linear 512->256<br/>LayerNorm + SiLU + Drop]:::dense
    FC1 -- (Batch, 256) --> FC2[Linear 256->Num_Classes]:::dense
    FC2 --> Out[Output Probabilities]:::input