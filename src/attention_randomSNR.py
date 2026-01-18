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
    # 数据集路径 (请确保这里指向的是包含信号数据的目录)
    # 如果你的原始 .npy 已经是加过噪声的，建议重新生成一份"纯净"数据，
    # 或者直接在现有数据上叠加更多噪声（虽然不严谨，但也能起到增强作用）。
    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"
    LOG_DIR = "./train_logs"

    # 训练超参数
    BATCH_SIZE = 64  # 48G显存推荐 64~128
    EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 4

    # === 新增：SNR 配置 ===
    SNR_MIN = 10  # 最小信噪比 dB
    SNR_MAX = 30  # 最大信噪比 dB

    # 自动获取的参数
    NUM_CLASSES = 0
    SAMPLE_LENGTH = 4096

    # 系统参数
    SAVE_INTERVAL = 5
    NUM_WORKERS = 0 if os.name == 'nt' else 4


config = Config()
os.makedirs(config.LOG_DIR, exist_ok=True)


# ===================== 2. 工具函数 (加噪核心) =====================
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


def add_awgn(signal, snr_db):
    """
    对信号添加高斯白噪声 (AWGN)
    input: signal (np.array) [2, L]
    input: snr_db (float) 信噪比
    output: noisy_signal (np.array) [2, L]
    """
    # 1. 计算信号功率 (P_signal)
    # 信号通常是复数形式 I+jQ，这里分开算的功率和是一样的
    # P = sum(x^2) / N
    signal_power = np.mean(np.sum(signal ** 2, axis=0))

    # 2. 根据 SNR 计算噪声功率 (P_noise)
    # SNR(dB) = 10 * log10(P_signal / P_noise)
    # => P_noise = P_signal / 10^(SNR/10)
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # 3. 生成噪声
    # 噪声需要分配到 I 和 Q 两路，所以单路功率要除以 2 (或者标准差除以 sqrt(2))
    noise_std = np.sqrt(noise_power / 2)
    noise = np.random.normal(0, noise_std, size=signal.shape)

    # 4. 叠加
    return signal + noise


# ===================== 3. 数据集 (支持在线随机加噪) =====================
class ModulationDataset(Dataset):
    def __init__(self, split='train', snr_range=(10, 30)):
        self.split = split
        self.snr_min, self.snr_max = snr_range

        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ {split}集文件缺失：{self.data_path}")

        try:
            # mmap_mode='r': 内存映射
            self.data = np.load(self.data_path, mmap_mode='r')
            self.labels = np.load(self.labels_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"❌ 加载{split}集失败：{e}")

        self.num_samples = len(self.labels)

        # 仅在训练集开启随机加噪，验证/测试集通常使用固定SNR或者也随机(视评估需求而定)
        # 这里默认全部随机，如果你希望测试集固定，可以在 getitem 里判断
        self.is_training = (split == 'train')

        log_info(f"✅ Loaded {split}: {self.num_samples:,} samples | SNR Mode: {self.snr_min}~{self.snr_max} dB")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # 1. 读取原始数据 (假设这里是纯净数据，或者低噪数据)
            # 复制一份出来以免修改 mmap 源文件
            sample_np = self.data[idx].astype(np.float32).copy()
            label_val = self.labels[idx]

            # 2. 在线加噪 (On-the-fly Augmentation)
            # 生成一个 [Min, Max] 之间的随机 SNR
            current_snr = np.random.uniform(self.snr_min, self.snr_max)

            # 调用加噪函数
            noisy_sample = add_awgn(sample_np, current_snr)

            return torch.from_numpy(noisy_sample.astype(np.float32)), torch.tensor(label_val, dtype=torch.long)

        except Exception as e:
            # 容错返回零张量
            return torch.zeros(2, config.SAMPLE_LENGTH).float(), torch.tensor(0).long()


# ===================== 4. 模型组件 (保持多域+注意力不变) =====================

class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()


class SEBlock1D(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.register_buffer('haar_weights', torch.tensor([
            [[0.70710678, 0.70710678]],
            [[0.70710678, -0.70710678]]
        ]).float())

    def forward(self, x):
        B, C, L = x.shape
        # FFT
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])
        fft_mag = torch.abs(torch.fft.fft(x_complex, dim=-1, norm='ortho'))
        fft_feature = torch.log1p(fft_mag).unsqueeze(1)

        # Wavelet
        x_reshaped = x.view(B * 2, 1, L)
        x_pad = F.pad(x_reshaped, (0, 1), mode='replicate')
        wavelet_out = F.conv1d(x_pad, self.haar_weights, stride=1)
        wavelet_feature = wavelet_out.view(B, 4, L)

        return torch.cat([x, fft_feature, wavelet_feature], dim=1)


class BLDE_1D_Modulation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = MultiDomainEmbedding()
        self.features = nn.Sequential(
            nn.Conv1d(7, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), get_activation(),
            SEBlock1D(32, reduction=4),

            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), get_activation(),
            SEBlock1D(64, reduction=8),
            nn.Dropout1d(0.1),

            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), get_activation(),
            SEBlock1D(128, reduction=16),

            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256), get_activation(),
            SEBlock1D(256, reduction=16),

            nn.Conv1d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), get_activation(),
            SEBlock1D(512, reduction=16),

            nn.Conv1d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), get_activation(),
            SEBlock1D(512, reduction=16),
        )
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
        x = self.embedding(x)
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ===================== 5. 训练主程序 =====================
def train_model():
    log_info("=" * 60)
    log_info(f"🚀 开始训练 | Random SNR: {config.SNR_MIN}~{config.SNR_MAX} dB | Multi-Domain + SE")
    log_info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取配置
    json_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            if 'label_to_idx' in mapping:
                config.NUM_CLASSES = len(mapping['label_to_idx'])
            elif isinstance(mapping, dict):
                config.NUM_CLASSES = len(mapping)
    else:
        config.NUM_CLASSES = 20
    log_info(f"📌 Classes: {config.NUM_CLASSES}")

    # --- 数据加载 (传递 SNR 参数) ---
    snr_range = (config.SNR_MIN, config.SNR_MAX)

    train_ds = ModulationDataset('train', snr_range=snr_range)
    # 验证集和测试集也保持随机SNR，以测试模型在动态环境下的鲁棒性
    # 如果想测试纯净信号，可修改为极高SNR，例如 (100, 100)
    val_ds = ModulationDataset('val', snr_range=snr_range)
    test_ds = ModulationDataset('test', snr_range=snr_range)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    # 模型初始化
    model = BLDE_1D_Modulation(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR,
        steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
        epochs=config.EPOCHS, pct_start=0.1
    )
    scaler = GradScaler()

    # 训练循环
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", ncols=110)

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

            scaler_loss = loss.item() * config.ACCUMULATION_STEPS
            total_loss += scaler_loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'Loss': f"{scaler_loss:.4f}", 'Acc': f"{100. * correct / total:.2f}%"})

        train_acc = 100. * correct / total
        log_info(
            f"Epoch {epoch + 1} Train | Loss: {total_loss / len(train_loader):.4f} | Acc: {train_acc:.2f}% | {monitor_resources()}")

        val_acc = evaluate(model, val_loader, device, criterion, "Val")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.DATASET_OUTPUT_DIR, "best_model.pth"))
            log_info(f"✅ Best Model Saved! Acc: {best_acc:.2f}%")

        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), os.path.join(config.DATASET_OUTPUT_DIR, f"epoch_{epoch + 1}.pth"))

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