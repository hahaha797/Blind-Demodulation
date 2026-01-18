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


# ===================== 1. 配置参数 =====================
class Config:
    # 数据集路径
    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"
    LOG_DIR = "./train_logs"

    # 训练超参数
    BATCH_SIZE = 64  # 根据显存调整 (48G显存可设 64~128)
    EPOCHS = 50
    LR = 3e-4  # 初始学习率
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 4  # 梯度累积，等效 Batch Size = 256

    # 自动获取的参数
    NUM_CLASSES = 0  # 稍后从 json 读取
    SAMPLE_LENGTH = 4096  # 样本长度

    # 系统参数
    SAVE_INTERVAL = 5
    NUM_WORKERS = 0 if os.name == 'nt' else 4


config = Config()
os.makedirs(config.LOG_DIR, exist_ok=True)


# ===================== 2. 工具函数 =====================
def log_info(msg, save_to_file=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    if save_to_file:
        with open(os.path.join(config.LOG_DIR, "train_log.txt"), 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")


def monitor_resources():
    mem_info = "CPU"
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
        mem_res = torch.cuda.memory_reserved(0) / 1024 ** 3
        mem_info = f"GPU: {mem_alloc:.1f}/{mem_res:.1f}GB"
    ram_used = psutil.virtual_memory().percent
    return f"{mem_info} | RAM: {ram_used}%"


# ===================== 3. 数据集 (Float16 Support) =====================
class ModulationDataset(Dataset):
    def __init__(self, split='train'):
        self.split = split
        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ {split}集文件缺失：{self.data_path}")

        try:
            # mmap_mode='r': 内存映射，不一次性加载进内存
            self.data = np.load(self.data_path, mmap_mode='r')
            self.labels = np.load(self.labels_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"❌ 加载{split}集失败：{e}")

        self.num_samples = len(self.labels)
        log_info(f"✅ Loaded {split}: {self.num_samples:,} samples | Shape: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # 读取数据并转为 Float32 (模型计算需要)
            sample_np = self.data[idx].astype(np.float32)
            label_val = self.labels[idx]
            return torch.from_numpy(sample_np), torch.tensor(label_val, dtype=torch.long)
        except:
            return torch.zeros(2, config.SAMPLE_LENGTH).float(), torch.tensor(0).long()


# ===================== 4. 模型组件 (SE-Block & Multi-Domain) =====================

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()


class SEBlock1D(nn.Module):
    """ 通道注意力机制：让模型学习“哪个通道更重要” """

    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        reduced_channel = max(channel // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channel, bias=False),
            get_activation(),
            nn.Linear(reduced_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiDomainEmbedding(nn.Module):
    """
    多域特征提取层
    将原始 I/Q 信号扩展为 [Time, FFT, Wavelet] 混合特征
    输出通道数: 2 (原始) + 1 (FFT) + 4 (Wavelet) = 7
    """

    def __init__(self):
        super().__init__()
        # Haar 小波核: Low-pass (均值/近似), High-pass (差分/细节)
        self.register_buffer('haar_weights', torch.tensor([
            [[0.70710678, 0.70710678]],  # Low
            [[0.70710678, -0.70710678]]  # High
        ]).float())  # Shape: [2, 1, 2]

    def forward(self, x):
        # Input x: [B, 2, L]
        B, C, L = x.shape

        # --- 1. 频域特征 (FFT) ---
        # 构造复数
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])
        # FFT 变换 -> 取模 -> 对数缩放 (log1p)
        fft_mag = torch.abs(torch.fft.fft(x_complex, dim=-1, norm='ortho'))
        fft_feature = torch.log1p(fft_mag).unsqueeze(1)  # [B, 1, L]

        # --- 2. 时频特征 (Wavelet) ---
        # 将 I, Q 视为独立通道处理
        x_reshaped = x.view(B * 2, 1, L)
        # Padding 保持卷积后长度不变 (L)
        x_pad = F.pad(x_reshaped, (0, 1), mode='replicate')
        # 卷积提取 Haar 特征
        wavelet_out = F.conv1d(x_pad, self.haar_weights, stride=1)  # [B*2, 2, L]
        # Reshape 回 [B, 4, L] (I-Low, I-High, Q-Low, Q-High)
        wavelet_feature = wavelet_out.view(B, 4, L)

        # --- 3. 拼接融合 ---
        # [B, 2+1+4, L] -> [B, 7, L]
        return torch.cat([x, fft_feature, wavelet_feature], dim=1)


class BLDE_1D_Modulation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 嵌入层
        self.embedding = MultiDomainEmbedding()

        # Backbone: Input Channels = 7
        self.features = nn.Sequential(
            # Stage 1: 7 -> 32
            nn.Conv1d(7, 32, 7, stride=2, padding=3, bias=False),  # -> 2048
            nn.BatchNorm1d(32), get_activation(),
            SEBlock1D(32, reduction=4),

            # Stage 2: 32 -> 64
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),  # -> 1024
            nn.BatchNorm1d(64), get_activation(),
            SEBlock1D(64, reduction=8),
            nn.Dropout1d(0.1),

            # Stage 3: 64 -> 128
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),  # -> 512
            nn.BatchNorm1d(128), get_activation(),
            SEBlock1D(128, reduction=16),

            # Stage 4: 128 -> 256
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False),  # -> 256
            nn.BatchNorm1d(256), get_activation(),
            SEBlock1D(256, reduction=16),

            # Stage 5: 256 -> 512
            nn.Conv1d(256, 512, 3, stride=2, padding=1, bias=False),  # -> 128
            nn.BatchNorm1d(512), get_activation(),
            SEBlock1D(512, reduction=16),

            # Stage 6: 512 -> 512
            nn.Conv1d(512, 512, 3, stride=2, padding=1, bias=False),  # -> 64
            nn.BatchNorm1d(512), get_activation(),
            SEBlock1D(512, reduction=16),
        )

        # Classifier Head
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
        x = self.embedding(x)  # [B, 7, L]
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ===================== 5. 训练主程序 =====================
def train_model():
    log_info("=" * 60)
    log_info("🚀 开始训练: Multi-Domain (Time+FFT+Wavelet) + SE-Attn + Float16")
    log_info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_info(f"Using device: {device}")

    # --- 1. 读取类别映射 ---
    json_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            if 'label_to_idx' in mapping:
                config.NUM_CLASSES = len(mapping['label_to_idx'])
            elif isinstance(mapping, dict):
                config.NUM_CLASSES = len(mapping)
        log_info(f"📌 Detected Classes: {config.NUM_CLASSES}")
    else:
        log_info("⚠️ label_mapping.json missing, defaulting to 20 classes")
        config.NUM_CLASSES = 20

    # --- 2. 加载数据 ---
    train_loader = DataLoader(ModulationDataset('train'), batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ModulationDataset('val'), batch_size=config.BATCH_SIZE * 2,
                            shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(ModulationDataset('test'), batch_size=config.BATCH_SIZE * 2,
                             shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # --- 3. 初始化模型 ---
    model = BLDE_1D_Modulation(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # One Cycle 学习率调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR,
        steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
        epochs=config.EPOCHS, pct_start=0.1
    )

    scaler = GradScaler()  # 混合精度

    # --- 4. 训练循环 ---
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", ncols=110)

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with autocast():  # 混合精度前向
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / config.ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            scaler_loss = loss.item() * config.ACCUMULATION_STEPS
            total_loss += scaler_loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'Loss': f"{scaler_loss:.4f}", 'Acc': f"{100. * correct / total:.2f}%"})

        # Epoch 结束统计
        train_acc = 100. * correct / total
        log_info(
            f"Epoch {epoch + 1} Train | Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc:.2f}% | {monitor_resources()}")

        # 验证
        val_acc = evaluate(model, val_loader, device, criterion, "Val")

        # 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.DATASET_OUTPUT_DIR, "best_model.pth"))
            log_info(f"✅ Best Model Saved! Acc: {best_acc:.2f}%")

        # 定期保存
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(config.DATASET_OUTPUT_DIR, f"epoch_{epoch + 1}.pth"))

    # 测试
    log_info("Starting Final Test...")
    model.load_state_dict(torch.load(os.path.join(config.DATASET_OUTPUT_DIR, "best_model.pth")))
    evaluate(model, test_loader, device, criterion, "Test")


def evaluate(model, loader, device, criterion, name="Val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
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
    torch.multiprocessing.freeze_support()
    train_model()