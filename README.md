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