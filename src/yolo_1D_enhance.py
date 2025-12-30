import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings
import psutil
from datetime import datetime

warnings.filterwarnings('ignore')


#

# ===================== 配置（适配 Float16 + 48G内存） =====================
class Config:
    # 1. 路径修改：指向生成的新数据集目录
    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"
    LOG_DIR = "./train_logs"

    # 2. 性能参数优化
    # 由于数据是Float16，内存占用减半，可以将Batch Size增大一倍
    BATCH_SIZE = 64  # 48G显存推荐64-128 (取决于模型大小)

    EPOCHS = 50  # 正式训练轮数
    LR = 3e-4  # 初始学习率 (AdamW)
    WEIGHT_DECAY = 1e-4

    # 梯度累积：Batch=64 * Accum=4 => 等效 Batch=256
    ACCUMULATION_STEPS = 4
    WARMUP_EPOCHS = 3

    # 3. 自动参数（后续从json读取）
    NUM_CLASSES = 0
    SAMPLE_LENGTH = 4096

    # 资源保护
    SAVE_INTERVAL = 5
    MAX_GPU_MEM_RATIO = 0.90


config = Config()

# 创建日志目录
os.makedirs(config.LOG_DIR, exist_ok=True)


# ===================== 工具函数 =====================
def log_info(msg, save_to_file=True):
    """打印并保存日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    if save_to_file:
        log_path = os.path.join(config.LOG_DIR, "train_log.txt")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")


def monitor_resources():
    """资源监控"""
    mem_info = "CPU"
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
        mem_res = torch.cuda.memory_reserved(0) / 1024 ** 3
        mem_info = f"GPU: {mem_alloc:.1f}/{mem_res:.1f}GB"

    ram_used = psutil.virtual_memory().percent
    return f"{mem_info} | RAM: {ram_used}%"


# ===================== 数据集类（适配 Float16 NPY） =====================
class ModulationDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ {split}集文件缺失：{self.data_path}")

        # === 核心优化：内存映射 ===
        # mmap_mode='r' 意味着数据保留在硬盘上，随用随取，不占用几十G的内存
        try:
            self.data = np.load(self.data_path, mmap_mode='r')
            self.labels = np.load(self.labels_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"❌ 加载{split}集失败：{e}")

        self.num_samples = len(self.labels)

        # 校验形状 [N, 2, L]
        if len(self.data.shape) != 3 or self.data.shape[1] != 2:
            log_info(f"⚠️ {split}集形状警告: {self.data.shape}, 预期 [N, 2, 4096]")

        log_info(f"✅ Loaded {split}: {self.num_samples:,} samples | Shape: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # === 核心适配：Float16 -> Float32 ===
            # 1. 从 mmap 读取 (Float16)
            # 2. astype(np.float32) 会将数据复制到内存并转换类型
            #    这是必要的，因为后续卷积层通常需要 float32 输入
            sample_np = self.data[idx].astype(np.float32)
            label_val = self.labels[idx]

            # 3. 转为 Tensor
            data_tensor = torch.from_numpy(sample_np)
            label_tensor = torch.tensor(label_val, dtype=torch.long)

            return data_tensor, label_tensor

        except Exception as e:
            # 容错处理
            return torch.zeros(2, config.SAMPLE_LENGTH).float(), torch.tensor(0).long()


# ===================== 模型定义 (BLDE-1D) =====================
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()


class BLDE_1D_Modulation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Backbone: Input [B, 2, 4096]
        # 结构未变，但输入数据现在是精准的 Float32 (由Dataset类转换而来)
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False),  # -> 2048
            nn.BatchNorm1d(32), get_activation(),

            # Stage 2
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),  # -> 1024
            nn.BatchNorm1d(64), get_activation(),
            nn.Dropout1d(0.1),

            # Stage 3
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),  # -> 512
            nn.BatchNorm1d(128), get_activation(),

            # Stage 4
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False),  # -> 256
            nn.BatchNorm1d(256), get_activation(),

            # Stage 5
            nn.Conv1d(256, 512, 3, stride=2, padding=1, bias=False),  # -> 128
            nn.BatchNorm1d(512), get_activation(),

            # Stage 6 (Global Context)
            nn.Conv1d(512, 512, 3, stride=2, padding=1, bias=False),  # -> 64
            nn.BatchNorm1d(512), get_activation(),
        )

        # Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            get_activation(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ===================== 训练主流程 =====================
def train_model():
    log_info("=" * 60)
    log_info("🚀 开始训练 (适配 Float16 数据集)")
    log_info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_info(f"Using device: {device}")

    # 1. 自动读取 Label Mapping 获取类别配置
    json_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                # 兼容两种json格式（之前代码生成的和直接字典的）
                if 'label_to_idx' in mapping:  # 旧格式
                    config.NUM_CLASSES = len(mapping['label_to_idx'])
                elif isinstance(mapping, dict):  # 新格式 (直接是字典)
                    config.NUM_CLASSES = len(mapping)
            log_info(f"📌 自动检测类别数: {config.NUM_CLASSES}")
        except Exception as e:
            log_info(f"⚠️ 读取json失败: {e}, 默认使用20类")
            config.NUM_CLASSES = 20
    else:
        log_info("⚠️ 未找到 label_mapping.json，默认使用20类")
        config.NUM_CLASSES = 20

    # 2. 数据集加载
    train_ds = ModulationDataset('train')
    val_ds = ModulationDataset('val')
    test_ds = ModulationDataset('test')

    # Windows下 num_workers 必须设为 0，否则 mmap 文件句柄会报错
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 3. 模型初始化
    model = BLDE_1D_Modulation(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 学习率策略 (OneCycle 效果通常最好)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR,
        steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
        epochs=config.EPOCHS, pct_start=0.1
    )

    scaler = GradScaler()  # 混合精度训练工具

    # 4. 训练循环
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", ncols=110)

        for i, (inputs, targets) in enumerate(pbar):
            # non_blocking=True 加速数据传输
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 混合精度上下文
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / config.ACCUMULATION_STEPS

            # 反向传播 (缩放梯度)
            scaler.scale(loss).backward()

            # 梯度累积更新
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # 统计
            scaler_loss = loss.item() * config.ACCUMULATION_STEPS
            total_loss += scaler_loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'Loss': f"{scaler_loss:.4f}", 'Acc': f"{100. * correct / total:.2f}%"})

        # End of Epoch
        train_acc = 100. * correct / total
        log_info(
            f"Epoch {epoch + 1} Train | Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc:.2f}% | {monitor_resources()}")

        # Validation
        val_acc = evaluate(model, val_loader, device, criterion, "Val")

        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.DATASET_OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            log_info(f"✅ New Best Model Saved! Acc: {best_acc:.2f}%")

        # Save Checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(config.DATASET_OUTPUT_DIR, f"epoch_{epoch + 1}.pth"))

    # Test
    log_info("Starting Final Test...")
    model.load_state_dict(torch.load(os.path.join(config.DATASET_OUTPUT_DIR, "best_model.pth")))
    evaluate(model, test_loader, device, criterion, "Test")


def evaluate(model, loader, device, criterion, name="Val"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=name, ncols=100, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    log_info(f"{name} Result | Loss: {total_loss / len(loader):.4f} | Acc: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    # Windows下防止多进程报错
    torch.multiprocessing.freeze_support()
    train_model()