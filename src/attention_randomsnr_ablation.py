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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ===================== 0. 字体配置函数 (新增) =====================
def setup_font():
    # 核心字体配置：英文/数字用Times New Roman，中文用宋体，跨系统兼容
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Times New Roman',
        'SimSun',
        'Songti SC',
        'DejaVu Serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # 字体大小设置：图例统一改为22号
    plt.rcParams['axes.titlesize'] = 52
    plt.rcParams['axes.labelsize'] = 52
    plt.rcParams['xtick.labelsize'] = 52
    plt.rcParams['ytick.labelsize'] = 52
    plt.rcParams['legend.fontsize'] = 20  # 🌟 修改：图例字号改为22
    plt.rcParams['figure.titlesize'] = 52

    print("✅ Font initialized: English=Times New Roman, Chinese=SimSun | Legend size=22")

# 立即调用字体配置，确保全局生效
setup_font()

# ===================== 1. 配置参数 =====================
class Config:
    # 🌟 核心模式控制 🌟
    # 1: 训练模式 (在混合 SNR 下训练所有的对比模型并保存最优权重)
    # 2: 验证绘图模式 (遍历不同 SNR，绘制宏观指标曲线和 20 种不重样的类间准确率曲线)
    # 3: 顶刊高级可视化模式 (遍历 4 个模型，在 0, 5, 10 dB 下绘制 12 张 t-SNE 和 混淆矩阵)
    MODE = 2

    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"
    LOG_DIR = "./train_logs"

    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    ACCUMULATION_STEPS = 4

    SNR_MIN = 10
    SNR_MAX = 30

    EVAL_SNRS = list(range(-10, 32, 2))

    # MODE 3: 高级可视化参数
    VISUALIZATION_SNRS = [0, 5, 10]  # 指定需要画图的信噪比列表
    TSNE_MAX_SAMPLES = 2000  # t-SNE 最大样本数，防止计算过久

    NUM_CLASSES = 0
    SAMPLE_LENGTH = 4096
    SAVE_INTERVAL = 5
    NUM_WORKERS = 0 if os.name == 'nt' else 4


config = Config()
os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(config.LOG_DIR, "plots"), exist_ok=True)


# ===================== 2. 工具函数 =====================
def log_info(msg, save_to_file=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)
    if save_to_file:
        with open(os.path.join(config.LOG_DIR, "train_log.txt"), 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")


def add_awgn(signal, snr_db):
    signal_power = np.mean(np.sum(signal ** 2, axis=0))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2)
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise


# ===================== 3. 数据集 =====================
class ModulationDataset(Dataset):
    def __init__(self, split='train', snr_range=(10, 30), fixed_snr=None):
        self.split = split
        self.snr_min, self.snr_max = snr_range
        self.fixed_snr = fixed_snr
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

            if self.fixed_snr is not None:
                current_snr = self.fixed_snr
            else:
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


class SEResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.use_se = use_se
        if self.use_se: self.se = SEBlock1D(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se: out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)


class FeatureBranch1D(nn.Module):
    def __init__(self, in_ch, use_se=False):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)

        self.layer1 = self._make_layer(32, 2, stride=2, use_se=use_se)
        self.layer2 = self._make_layer(64, 2, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(128, 2, stride=2, use_se=use_se)
        self.layer4 = self._make_layer(256, 2, stride=2, use_se=use_se)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, out_channels, num_blocks, stride, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SEResBlock1D(self.in_channels, out_channels, s, use_se))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)


# ----------------- 核心模型 A, B -----------------
class MultiBranchFusionNet(nn.Module):
    def __init__(self, num_classes, use_se=False):
        super().__init__()
        self.register_buffer('haar_weights', torch.tensor([
            [[0.70710678, 0.70710678]], [[0.70710678, -0.70710678]]
        ]).float())

        self.branch_iq = FeatureBranch1D(in_ch=2, use_se=use_se)
        self.branch_fft = FeatureBranch1D(in_ch=1, use_se=use_se)
        self.branch_wave = FeatureBranch1D(in_ch=4, use_se=use_se)

        fusion_dim = 256 * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), get_activation(), nn.Dropout(0.4),
            nn.Linear(512, 256), get_activation(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def get_features(self, x):
        B, C, L = x.shape
        iq_data = x
        x_complex = torch.complex(x[:, 0, :], x[:, 1, :])
        fft_mag = torch.abs(torch.fft.fft(x_complex, dim=-1, norm='ortho'))
        fft_data = torch.log1p(fft_mag).unsqueeze(1)
        x_reshaped = x.view(B * 2, 1, L)
        x_pad = F.pad(x_reshaped, (0, 1), mode='replicate')
        wavelet_out = F.conv1d(x_pad, self.haar_weights, stride=1)
        wave_data = wavelet_out.view(B, 4, L)
        return iq_data, fft_data, wave_data

    def forward(self, x, return_features=False):
        iq_data, fft_data, wave_data = self.get_features(x)
        feat_iq = self.branch_iq(iq_data)
        feat_fft = self.branch_fft(fft_data)
        feat_wave = self.branch_wave(wave_data)
        fused_features = torch.cat([feat_iq, feat_fft, feat_wave], dim=1)
        out = self.classifier(fused_features)

        if return_features: return out, fused_features
        return out


# ----------------- 基线模型 C (纯 ResNet) -----------------
class PureResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.branch_iq = FeatureBranch1D(in_ch=2, use_se=False)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), get_activation(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    # 🌟 修改：支持 return_features
    def forward(self, x, return_features=False):
        features = self.branch_iq(x)
        out = self.classifier(features)
        if return_features: return out, features
        return out


# ----------------- 基线模型 D (CNN-LSTM) -----------------
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    # 🌟 修改：支持 return_features
    def forward(self, x, return_features=False):
        x = self.cnn(x).permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(x)
        hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        out = self.fc(hidden)
        if return_features: return out, hidden
        return out


# ===================== 5. 核心引擎 (三种模式) =====================
def get_model_dict():
    return {
        "A_Proposed_MultiBranch_SE": lambda c: MultiBranchFusionNet(c, use_se=True),
        "B_Ablation_MultiBranch_NoSE": lambda c: MultiBranchFusionNet(c, use_se=False),
        "C_Baseline_Pure_ResNet_IQ": lambda c: PureResNet1D(c),
        "D_Baseline_CNN_LSTM": lambda c: CNN_LSTM(c)
    }


# ----- MODE 1: 训练 -----
def run_training_mode(device, criterion):
    # ...(略，与之前完全相同)
    log_info("=== 🚀 正在进入 MODE 1: 训练模式 ===")
    snr_range = (config.SNR_MIN, config.SNR_MAX)
    train_loader = DataLoader(ModulationDataset('train', snr_range), batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ModulationDataset('val', snr_range), batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    def train_single_model(model_name, model_func):
        model = model_func(config.NUM_CLASSES).to(device)
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
                pbar.set_postfix({'Loss': f"{loss.item() * config.ACCUMULATION_STEPS:.4f}"})

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            val_acc = 100. * correct / total
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                log_info(f"[{model_name}] ✅ 验证集精度提升至 {best_acc:.2f}%, 模型已保存")

    for name, func in get_model_dict().items():
        log_info(f"\n" + "=" * 50)
        log_info(f"🚀 开始训练: {name}")
        train_single_model(name, func)


# ----- MODE 2: 验证绘图 -----
def run_evaluation_mode(device, class_names):
    # ...(略，与之前完全相同)
    log_info("=== 🧪 正在进入 MODE 2: 测试与绘图模式 ===")
    models_to_eval = get_model_dict()
    metrics = {m: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for m in models_to_eval.keys()}
    per_class = {m: {c: [] for c in class_names} for m in models_to_eval.keys()}

    for model_name, model_func in models_to_eval.items():
        model_save_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{model_name}_best.pth")
        if not os.path.exists(model_save_path):
            continue
        model = model_func(config.NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        for snr in config.EVAL_SNRS:
            test_ds = ModulationDataset('test', fixed_snr=snr)
            test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE * 2, num_workers=config.NUM_WORKERS)
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            metrics[model_name]['acc'].append(accuracy_score(all_targets, all_preds))
            p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
            metrics[model_name]['prec'].append(p)
            metrics[model_name]['rec'].append(r)
            metrics[model_name]['f1'].append(f1)

            cm = confusion_matrix(all_targets, all_preds, labels=range(config.NUM_CLASSES))
            class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
            for i, c_name in enumerate(class_names):
                per_class[model_name][c_name].append(class_accuracies[i])

    snrs = config.EVAL_SNRS
    plot_dir = os.path.join(config.LOG_DIR, "plots")

    metric_titles = {'acc': 'Accuracy', 'prec': 'Macro Precision', 'rec': 'Macro Recall', 'f1': 'Macro F1-Score'}
    for m_key, m_name in metric_titles.items():
        plt.figure(figsize=(15, 10))
        for model_name in models_to_eval.keys():
            if len(metrics[model_name][m_key]) > 0:
                plt.plot(snrs, metrics[model_name][m_key], marker='o', label=model_name, linewidth=2)
        plt.title(f'{m_name} vs. SNR', fontsize=52)
        plt.xlabel('SNR (dB)', fontsize=52)
        plt.ylabel(m_name, fontsize=52)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()  # 使用全局配置的22号字体
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"Overall_{m_key}.png"), dpi=300)
        plt.close()

    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '>']
    for model_name in models_to_eval.keys():
        if len(per_class[model_name][class_names[0]]) == 0: continue
        plt.figure(figsize=(16, 12))
        for i, c_name in enumerate(class_names):
            plt.plot(snrs, per_class[model_name][c_name], marker=markers[i % len(markers)],
                     linestyle=line_styles[i % len(line_styles)], color=colors[i], label=c_name, linewidth=2.5)
        plt.title(f'Per-Class Accuracy vs. SNR\nModel: {model_name}', fontsize=52)
        plt.xlabel('SNR (dB)', fontsize=52)
        plt.ylabel('Accuracy', fontsize=52)
        plt.grid(True, linestyle='--', alpha=0.5)

        # ====================== 核心修改：图例放进图内 ======================
        plt.legend(loc='lower right', ncol=2, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"PerClass_Acc_{model_name}.png"), dpi=300)
        plt.close()
    log_info(f"✅ MODE 2 图表保存完毕: {plot_dir}")


# ----------------- MODE 3: 顶刊高级可视化 -----------------
def run_visualization_mode(device, class_names):
    log_info(f"=== 🎨 正在进入 MODE 3: 顶刊高级可视化模式 ===")

    target_snrs = config.VISUALIZATION_SNRS
    models_to_eval = get_model_dict()
    plot_dir = os.path.join(config.LOG_DIR, "plots")
    colors = plt.cm.tab20(np.linspace(0, 1, config.NUM_CLASSES))

    # 第一层循环：遍历所有 4 个模型
    for model_name, model_func in models_to_eval.items():
        model_save_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{model_name}_best.pth")

        if not os.path.exists(model_save_path):
            log_info(f"⚠️ 找不到权重 {model_save_path}，跳过 {model_name}。")
            continue

        model = model_func(config.NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        # 第二层循环：遍历目标 SNR [0, 5, 10]
        for snr in target_snrs:
            log_info(f"[{model_name}] 正在提取 SNR={snr}dB 的深层特征...")

            test_ds = ModulationDataset('test', fixed_snr=snr)
            test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=True)

            all_preds, all_targets, all_features = [], [], []

            with torch.no_grad():
                for inputs, targets in tqdm(test_loader, leave=False, ncols=100):
                    inputs = inputs.to(device)
                    # 所有的基线模型现在也都支持 return_features=True 了
                    outputs, features = model(inputs, return_features=True)
                    _, predicted = outputs.max(1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    # 将高维特征展平并保存
                    all_features.append(features.view(features.size(0), -1).cpu().numpy())

                    if len(all_targets) >= config.TSNE_MAX_SAMPLES:
                        break

            all_features = np.concatenate(all_features, axis=0)[:config.TSNE_MAX_SAMPLES]
            all_targets = np.array(all_targets[:config.TSNE_MAX_SAMPLES])
            all_preds = np.array(all_preds[:config.TSNE_MAX_SAMPLES])

            # ---------------- 1. 绘制 混淆矩阵热力图 ----------------
            cm = confusion_matrix(all_targets, all_preds, labels=range(config.NUM_CLASSES))
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

            plt.figure(figsize=(16, 14))
            sns.heatmap(cm_normalized, annot=False, cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=.5, linecolor='gray', vmin=0, vmax=1)
            plt.title(f'Confusion Matrix at {snr}dB\nModel: {model_name}', fontsize=52)
            plt.xlabel('Predicted Label', fontsize=52)
            plt.ylabel('True Label', fontsize=52)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"CM_{model_name}_{snr}dB.png"), dpi=300)
            plt.close()

            # ---------------- 2. 绘制 t-SNE 聚类图 ----------------
            log_info(f"[{model_name} @ {snr}dB] 正在进行 t-SNE 降维计算...")
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
            tsne_results = tsne.fit_transform(all_features)

            plt.figure(figsize=(16, 12))
            for i in range(config.NUM_CLASSES):
                idx = np.where(all_targets == i)[0]
                if len(idx) > 0:
                    plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                                c=[colors[i]], label=class_names[i],
                                alpha=0.8, edgecolors='w', s=60)

            plt.title(f't-SNE Feature Visualization at {snr}dB\nModel: {model_name}', fontsize=52)
            plt.xticks([])
            plt.yticks([])
            # 🌟 移除手动设置的fontsize，使用全局配置的22号字体
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"t-SNE_{model_name}_{snr}dB.png"), dpi=300)
            plt.close()

    log_info(f"✅ MODE 3 所有模型的 t-SNE 与 混淆矩阵保存完毕: {plot_dir}")


# ===================== 6. 主程序入口 =====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    json_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    class_names = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            if 'label_to_idx' in mapping:
                label_to_idx = mapping['label_to_idx']
            else:
                label_to_idx = mapping

            config.NUM_CLASSES = len(label_to_idx)
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            class_names = [idx_to_label[i] for i in range(config.NUM_CLASSES)]
    else:
        config.NUM_CLASSES = 20
        class_names = [f"Class_{i}" for i in range(20)]

    log_info(f"📌 系统设置 | 模式: MODE {config.MODE} | Classes: {config.NUM_CLASSES} | Device: {device}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if config.MODE == 1:
        run_training_mode(device, criterion)
    elif config.MODE == 2:
        run_evaluation_mode(device, class_names)
    elif config.MODE == 3:
        run_visualization_mode(device, class_names)
    else:
        log_info("❌ 错误的 MODE 设置，请设置为 1, 2 或 3。")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()