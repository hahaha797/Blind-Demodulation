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
    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"
    LOG_DIR = "./train_logs"

    BATCH_SIZE = 64
    EPOCHS = 10  # 为了对比测试，建议先将 Epoch 调小快速验证
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 4

    SNR_MIN = 10
    SNR_MAX = 30

    NUM_CLASSES = 0
    SAMPLE_LENGTH = 4096
    SAVE_INTERVAL = 10
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


def add_awgn(signal, snr_db):
    signal_power = np.mean(np.sum(signal ** 2, axis=0))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2)
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise


# ===================== 3. 数据集 =====================
class ModulationDataset(Dataset):
    def __init__(self, split='train', snr_range=(10, 30)):
        self.split = split
        self.snr_min, self.snr_max = snr_range
        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"❌ {split}集文件缺失：{self.data_path}")

        self.data = np.load(self.data_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')
        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            sample_np = self.data[idx].astype(np.float32).copy()
            label_val = self.labels[idx]
            current_snr = np.random.uniform(self.snr_min, self.snr_max)
            noisy_sample = add_awgn(sample_np, current_snr)
            return torch.from_numpy(noisy_sample.astype(np.float32)), torch.tensor(label_val, dtype=torch.long)
        except Exception:
            return torch.zeros(2, config.SAMPLE_LENGTH).float(), torch.tensor(0).long()


# ===================== 4. 模型组件与定义 =====================
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
            nn.Linear(channel, reduced_channel, bias=False), get_activation(),
            nn.Linear(reduced_channel, channel, bias=False), nn.Sigmoid()
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
            [[0.70710678, 0.70710678]], [[0.70710678, -0.70710678]]
        ]).float())

    def forward(self, x):
        B, C, L = x.shape
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])
        fft_mag = torch.abs(torch.fft.fft(x_complex, dim=-1, norm='ortho'))
        fft_feature = torch.log1p(fft_mag).unsqueeze(1)
        x_reshaped = x.view(B * 2, 1, L)
        x_pad = F.pad(x_reshaped, (0, 1), mode='replicate')
        wavelet_out = F.conv1d(x_pad, self.haar_weights, stride=1)
        wavelet_feature = wavelet_out.view(B, 4, L)
        return torch.cat([x, fft_feature, wavelet_feature], dim=1)


# --- 模型 A: 你的原始模型 (包含多域特征提取和 SE) ---
class BLDE_1D_Modulation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = MultiDomainEmbedding()
        self.features = nn.Sequential(
            nn.Conv1d(7, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), get_activation(),
            SEBlock1D(32, 4),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), get_activation(),
            SEBlock1D(64, 8), nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(128), get_activation(),
            SEBlock1D(128, 16),
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(256), get_activation(),
            SEBlock1D(256, 16),
            nn.Conv1d(256, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(512), get_activation(),
            SEBlock1D(512, 16),
            nn.Conv1d(512, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(512), get_activation(),
            SEBlock1D(512, 16),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.LayerNorm(256), get_activation(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(self.embedding(x))))


# --- 模型 B: 消融模型 (去除多域特征和 SE) ---
class Ablation_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # 注意：输入通道恢复为 2，不使用 SEBlock
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), get_activation(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), get_activation(),
            nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(128), get_activation(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(256), get_activation(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(512), get_activation(),
            nn.Conv1d(512, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(512), get_activation(),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.LayerNorm(256), get_activation(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


# --- 模型 C: 常见基准算法 ResNet-1D ---
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 2, stride=2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# --- 模型 D: 常见基准算法 CNN-LSTM ---
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 用 CNN 降采样序列长度，避免 LSTM 直接处理 4096 过慢
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),  # Bidirectional = 128 * 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # [B, 128, L/8]
        x = x.permute(0, 2, 1)  # 变幻维度供 LSTM 使用: [B, L/8, 128]
        lstm_out, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出，处理双向
        hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        return self.fc(hidden)


# ===================== 5. 训练与评估引擎 =====================
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
    return 100. * correct / total


def train_single_model(model_name, model_class, train_loader, val_loader, test_loader, device, criterion):
    log_info(f"\n" + "=" * 50)
    log_info(f"🚀 开始训练模型: {model_name}")
    log_info("=" * 50)

    model = model_class(num_classes=config.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.LR, steps_per_epoch=len(train_loader) // config.ACCUMULATION_STEPS,
        epochs=config.EPOCHS, pct_start=0.1
    )
    scaler = GradScaler()
    best_acc = 0.0
    model_save_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{model_name}_best.pth")

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS} [{model_name}]", ncols=110)

        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets) / config.ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * config.ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'Loss': f"{loss.item() * config.ACCUMULATION_STEPS:.4f}"})

        val_acc = evaluate(model, val_loader, device, criterion, "Val")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            log_info(f"[{model_name}] ✅ 验证集精度提升至 {best_acc:.2f}%, 模型已保存")

    # 测试最终效果
    model.load_state_dict(torch.load(model_save_path))
    test_acc = evaluate(model, test_loader, device, criterion, "Test")
    log_info(f"🎉 [{model_name}] 最终测试集精度: {test_acc:.2f}%")
    return test_acc


# ===================== 6. 主程序入口 =====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 动态读取类别数
    json_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            config.NUM_CLASSES = len(mapping.get('label_to_idx', mapping))
    else:
        config.NUM_CLASSES = 20

    log_info(
        f"📌 系统设置 | Classes: {config.NUM_CLASSES} | SNR: {config.SNR_MIN}~{config.SNR_MAX} dB | Device: {device}")

    # 数据加载器 (所有模型共用相同的数据分布)
    snr_range = (config.SNR_MIN, config.SNR_MAX)
    train_loader = DataLoader(ModulationDataset('train', snr_range), batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ModulationDataset('val', snr_range), batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(ModulationDataset('test', snr_range), batch_size=config.BATCH_SIZE * 2, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 定义要对比的模型列表
    models_to_train = {
        "A_BLDE_Proposed": BLDE_1D_Modulation,
        "B_Ablation_CNN": Ablation_CNN,
        "C_ResNet1D": ResNet1D,
        "D_CNN_LSTM": CNN_LSTM
    }

    results = {}

    # 循环训练每一个模型
    for model_name, model_class in models_to_train.items():
        test_acc = train_single_model(model_name, model_class, train_loader, val_loader, test_loader, device, criterion)
        results[model_name] = test_acc

    # 打印最终对比结果
    log_info("\n" + "=" * 60)
    log_info("📊 最终多模型对比结果汇总:")
    log_info("=" * 60)
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        log_info(f" 🏆 {model_name.ljust(20)}: {acc:.2f}%")
    log_info("=" * 60)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()