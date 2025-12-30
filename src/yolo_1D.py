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


# ===================== 配置（适配 Float16 数据集） =====================
class Config:
    # 指向新的数据集目录
    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"

    # 训练日志目录
    LOG_DIR = "./train_logs"

    # === 训练超参数 ===
    # 由于使用了Float16数据，显存占用更小，可以尝试增大Batch Size
    BATCH_SIZE = 64  # 48G显存推荐64或128
    EPOCHS = 50  # 总轮数
    LR = 3e-4  # 初始学习率 (AdamW)
    WEIGHT_DECAY = 1e-4  # 权重衰减

    # 梯度累积：如果显存不够，增大这个值，减小BatchSize
    # 现在的 batch=64, accum=4 等效于 batch=256
    ACCUMULATION_STEPS = 4

    WARMUP_EPOCHS = 3  # 预热轮数

    # === 自动填充（不要手动改，会从json读取） ===
    NUM_CLASSES = 0
    SAMPLE_LENGTH = 4096

    # 资源保护
    SAVE_INTERVAL = 5
    MAX_GPU_MEM_RATIO = 0.90


config = Config()

# 创建目录
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
    """简单的资源监控"""
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
        mem_res = torch.cuda.memory_reserved(0) / 1024 ** 3
        return f"GPU: {mem_alloc:.1f}/{mem_res:.1f}GB"
    return "CPU"


# ===================== 数据集类（适配 .npy Float16） =====================
class ModulationDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ {split}集文件缺失：{self.data_path}")

        # 使用 mmap_mode='r' 实现内存映射，不一次性加载到内存
        # 即使文件很大，系统内存占用也会很低
        try:
            self.data = np.load(self.data_path, mmap_mode='r')
            self.labels = np.load(self.labels_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"❌ 加载{split}集失败：{e}")

        self.num_samples = len(self.labels)

        # 校验形状 [N, 2, L]
        if len(self.data.shape) != 3 or self.data.shape[1] != 2:
            log_info(f"⚠️ {split}集形状可能不匹配: {self.data.shape}, 预期 [N, 2, 4096]")

        log_info(f"✅ Loaded {split}: {self.num_samples:,} samples | Shape: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # 1. 读取数据 (从硬盘/缓存读取 Float16)
            # copy() 是为了将 mmap 的只读数据转为内存中的可写副本，避免Torch报错
            sample_np = self.data[idx].copy()
            label_val = self.labels[idx]

            # 2. 转换为 Tensor 并转为 Float32
            # 虽然存的是 Float16，但进入模型通常需要 Float32 (除非全流程Half)
            data_tensor = torch.from_numpy(sample_np).float()
            label_tensor = torch.tensor(label_val, dtype=torch.long)

            return data_tensor, label_tensor

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return torch.zeros(2, config.SAMPLE_LENGTH).float(), torch.tensor(0).long()


# ===================== 模型定义 (YOLO12-1D) =====================
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()


class YOLO12_1D_Modulation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Backbone: Input [B, 2, 4096]
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
    log_info("🚀 开始训练 (适配新数据集)")
    log_info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_info(f"Using device: {device}")

    # 1. 读取 Label Mapping 获取类别配置
    json_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # 兼容两种json格式（之前代码生成的和直接字典的）
            if 'label_to_idx' in mapping:
                config.NUM_CLASSES = len(mapping['label_to_idx'])
            else:
                config.NUM_CLASSES = len(mapping)
        log_info(f"📌 自动检测类别数: {config.NUM_CLASSES}")
    else:
        log_info("⚠️ 未找到 label_mapping.json，默认使用20类")
        config.NUM_CLASSES = 20

    # 2. 数据集加载
    train_ds = ModulationDataset('train')
    val_ds = ModulationDataset('val')
    test_ds = ModulationDataset('test')

    # Windows下 num_workers 建议设为 0，避免多进程与 mmap 冲突
    # Linux下 可以设为 4 或 8
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # 3. 模型初始化
    model = YOLO12_1D_Modulation(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 简单的学习率衰减
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR,
        steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
        epochs=config.EPOCHS, pct_start=0.1
    )

    scaler = GradScaler()  # 混合精度训练

    # 4. 循环
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", ncols=100)

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / config.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

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
    # Windows下如果报错 broken pipe，保留这行；否则在主函数外执行
    torch.multiprocessing.freeze_support()
    train_model()