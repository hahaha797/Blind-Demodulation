import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ===================== 配置（仅修改此处） =====================
class Config:
    DATASET_OUTPUT_DIR = "./modulation_dataset"  # 数据集构造程序输出目录
    BATCH_SIZE = 8  # 8GB显存适配（12GB可设16，24GB可设32）
    EPOCHS = 5  # 测试用，千万级样本建议正式训练设50-100
    LR = 1e-4  # 千万级样本建议学习率1e-4~5e-5
    WEIGHT_DECAY = 1e-5  # 防止过拟合
    ACCUMULATION_STEPS = 4  # 梯度累积，等效增大批次（48G内存建议4-8）
    # 自动从label_mapping.json读取类别数，无需手动设置
    NUM_CLASSES = None


config = Config()


# ===================== 数据集类（适配千万级样本+48G内存） =====================
class LargeNpyDataset(Dataset):
    """
    内存映射模式加载大npy文件，避免一次性加载到内存
    适配千万级样本、48G内存场景，仅按需读取样本
    """

    def __init__(self, split='train'):
        self.split = split
        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        # 内存映射模式加载（核心优化：不占物理内存）
        self.data = np.load(self.data_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')

        # 打印数据集信息（适配你的12类调制信号）
        self.num_samples = len(self.data)
        self.unique_labels = np.unique(self.labels[:10000])  # 采样统计类别数
        print(f"✅ 加载{split}集：{self.num_samples:,}个样本 | 检测到{len(self.unique_labels)}类调制信号")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 按需读取单样本，避免内存溢出
        try:
            data = torch.from_numpy(self.data[idx]).float()
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return data, label
        except Exception as e:
            # 容错：样本读取失败时返回空样本（避免训练中断）
            print(f"⚠️  读取{self.split}集样本{idx}失败：{e}")
            return torch.zeros(2, config.SAMPLE_LENGTH).float(), torch.tensor(0, dtype=torch.long)


# ===================== 模型定义（兼容12类调制信号） =====================
class Swish(nn.Module):
    """低版本PyTorch兼容SiLU"""

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()


class YOLO12_1D_Classifier(nn.Module):
    """轻量化YOLO12-1D模型（适配12类调制信号分类）"""

    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(2, 16, 6, 2, 3, bias=False),  # 输入：2通道（I/Q），4096长度
            nn.BatchNorm1d(16),
            get_activation(),
            nn.Conv1d(16, 32, 3, 2, 1, bias=False),
            nn.BatchNorm1d(32),
            get_activation(),
            nn.Conv1d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            get_activation(),
            nn.Conv1d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            get_activation(),
            nn.Conv1d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            get_activation(),
        )
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 适配任意长度输入
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            get_activation(),
            nn.Dropout(0.1),  # 千万级样本建议dropout 0.1-0.2
            nn.Linear(64, num_classes)
        )
        # 权重初始化（适配分类任务）
        self.apply(lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if isinstance(m, (nn.Conv1d, nn.Linear)) else None)
        # 自动适配设备（GPU/CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.class_head(self.backbone(x.to(self.device)))


# ===================== 训练函数（适配千万级样本） =====================
def train_model():
    """主训练函数：适配12类调制信号、千万级样本、48G内存"""
    print("\n" + "=" * 80)
    print("🚀 开始训练（适配千万级调制信号数据集）")
    print("=" * 80)

    # -------------------------- 1. 初始化配置（自动适配你的数据集） --------------------------
    # 自动读取样本长度（从file_metadata.csv或label_mapping.json）
    try:
        label_mapping_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        config.NUM_CLASSES = len(label_mapping['label_to_idx'])
        config.SAMPLE_LENGTH = 4096  # 你的数据集固定4096长度
        print(f"📌 自动识别：{config.NUM_CLASSES}类调制信号 | 单样本长度{config.SAMPLE_LENGTH}")
    except Exception as e:
        print(f"⚠️  读取标签映射失败，使用默认12类：{e}")
        config.NUM_CLASSES = 12
        config.SAMPLE_LENGTH = 4096

    # 校验数据集文件（适配你的流式生成结果）
    required_npy = [
        "train_data.npy", "train_labels.npy",
        "val_data.npy", "val_labels.npy",
        "test_data.npy", "test_labels.npy"
    ]
    missing_files = []
    for f in required_npy:
        file_path = os.path.join(config.DATASET_OUTPUT_DIR, f)
        if not os.path.exists(file_path):
            missing_files.append(f)
    if missing_files:
        raise FileNotFoundError(f"❌ 缺失数据集文件：{missing_files}，请先运行dataset_constructor.py")

    # -------------------------- 2. 设备适配（Windows+48G内存） --------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # 显存分片，避免OOM
    torch.backends.cudnn.benchmark = True  # 加速训练
    torch.multiprocessing.set_sharing_strategy('file_system')  # Windows内存映射兼容
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📌 训练设备：{device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    if not torch.cuda.is_available():
        print("⚠️  未检测到GPU，将使用CPU训练（千万级样本CPU训练极慢）")

    # -------------------------- 3. 加载数据集（内存映射模式） --------------------------
    print("\n📌 加载数据集（内存映射模式，48G内存友好）")
    train_dataset = LargeNpyDataset('train')
    val_dataset = LargeNpyDataset('val')
    test_dataset = LargeNpyDataset('test')

    # DataLoader（适配千万级样本，关闭多进程避免内存冲突）
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,  # num_workers=0适配Windows
        prefetch_factor=None  # 关闭预取，减少内存占用
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # -------------------------- 4. 模型/优化器初始化 --------------------------
    model = YOLO12_1D_Classifier(num_classes=config.NUM_CLASSES)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        eps=1e-8  # 适配千万级样本优化
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6  # 学习率余弦衰减
    )
    scaler = GradScaler()  # 混合精度训练，节省显存

    # -------------------------- 5. 训练循环（适配千万级样本） --------------------------
    best_val_acc = 0.0
    print(f"\n📌 开始训练：{config.EPOCHS}轮 | 批次大小{config.BATCH_SIZE} | 梯度累积{config.ACCUMULATION_STEPS}")

    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss, train_acc, train_total = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config.EPOCHS}] Train")

        optimizer.zero_grad()
        for batch_idx, (data, labels) in enumerate(pbar):
            # 定期清理显存（适配千万级样本）
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            # 混合精度训练（核心：减少显存占用）
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels.to(device)) / config.ACCUMULATION_STEPS

            # 梯度累积（等效增大批次，提升训练稳定性）
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 统计训练指标
            train_loss += loss.item() * config.ACCUMULATION_STEPS * data.size(0)
            train_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
            train_total += data.size(0)

            # 实时显示显存占用
            mem_used = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'Loss': f'{loss.item() * config.ACCUMULATION_STEPS:.4f}',
                'Acc': f'{train_acc / train_total:.4f}',
                'Mem': f'{mem_used:.1f}GB'
            })

        # 验证阶段（无梯度计算）
        model.eval()
        val_loss, val_acc, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{config.EPOCHS}] Val"):
                outputs = model(data)
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item() * data.size(0)
                val_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
                val_total += data.size(0)

        # 计算本轮指标
        train_loss_avg = train_loss / train_total
        train_acc_avg = train_acc / train_total
        val_loss_avg = val_loss / val_total
        val_acc_avg = val_acc / val_total
        scheduler.step()  # 学习率衰减

        # 打印本轮结果
        print(f"\n📊 Epoch {epoch + 1} 训练结果：")
        print(f"  - 训练损失：{train_loss_avg:.4f} | 训练准确率：{train_acc_avg:.4f}")
        print(f"  - 验证损失：{val_loss_avg:.4f} | 验证准确率：{val_acc_avg:.4f}")
        print(f"  - 当前学习率：{optimizer.param_groups[0]['lr']:.6f}")

        # 保存最优模型（按验证准确率）
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            save_path = os.path.join(config.DATASET_OUTPUT_DIR, "yolo12_1d_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'num_classes': config.NUM_CLASSES,
                'sample_length': config.SAMPLE_LENGTH
            }, save_path)
            print(f"✅ 保存最优模型：{save_path}（验证准确率：{best_val_acc:.4f}）")

    # -------------------------- 6. 测试阶段（加载最优模型） --------------------------
    print("\n📌 测试阶段（加载最优模型评估）")
    checkpoint = torch.load(os.path.join(config.DATASET_OUTPUT_DIR, "yolo12_1d_best.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_acc, test_total = 0.0, 0
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing"):
            outputs = model(data)
            test_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
            test_total += data.size(0)
    test_acc_avg = test_acc / test_total

    # -------------------------- 7. 最终结果输出 --------------------------
    print("\n" + "=" * 80)
    print("🎉 训练完成！最终结果（适配千万级调制信号数据集）：")
    print(f"  - 最优验证准确率：{best_val_acc:.4f}")
    print(f"  - 测试集准确率：{test_acc_avg:.4f}")
    print(f"  - 调制类别数：{config.NUM_CLASSES}")
    print(f"  - 最优模型路径：{os.path.join(config.DATASET_OUTPUT_DIR, 'yolo12_1d_best.pth')}")
    print("=" * 80)


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 内存保护：限制PyTorch内存占用（48G内存建议设为32G）
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # 最多使用80%GPU显存
    train_model()