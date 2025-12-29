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

# ===================== 配置 =====================
class Config:
    DATASET_OUTPUT_DIR = "./modulation_dataset"
    BATCH_SIZE = 8  # 8GB显存适配
    EPOCHS = 5
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    ACCUMULATION_STEPS = 4
    # 自动读取调制类型数
    NUM_CLASSES = len(json.load(open(os.path.join(DATASET_OUTPUT_DIR, "label_mapping.json")))['label_to_idx'])

config = Config()

# ===================== 流式数据集加载类 =====================
class StreamNpyDataset(Dataset):
    """兼容流式生成的npy数据集"""
    def __init__(self, split='train'):
        self.data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        self.labels_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")
        # 内存映射加载（避免一次性加载大文件）
        self.data = np.load(self.data_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')
        print(f"✅ 加载{split}集：{len(self.data)}个样本（内存映射模式）")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 按需读取，不缓存
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx], dtype=torch.long)

# ===================== 模型定义 =====================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()

class YOLO12_1D_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(2, 16, 6, 2, 3, bias=False),
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
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            get_activation(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        self.apply(lambda m: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                   if isinstance(m, (nn.Conv1d, nn.Linear)) else None)
        self.to('cuda')

    def forward(self, x):
        return self.class_head(self.backbone(x.to('cuda')))

# ===================== 训练函数 =====================
def train_model():
    print("\n" + "=" * 80)
    print("🚀 开始训练（兼容流式数据集）")
    print("=" * 80)

    # 校验数据集
    required_files = [f"{split}_data.npy" for split in ['train', 'val', 'test']] + \
                     [f"{split}_labels.npy" for split in ['train', 'val', 'test']]
    for f in required_files:
        assert os.path.exists(os.path.join(config.DATASET_OUTPUT_DIR, f)), f"❌ 找不到{f}！请先运行dataset_constructor.py"

    # Windows适配
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📌 设备：{device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 加载数据集（内存映射模式）
    train_dataset = StreamNpyDataset('train')
    val_dataset = StreamNpyDataset('val')
    test_dataset = StreamNpyDataset('test')

    # DataLoader（多进程关闭，避免内存映射冲突）
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False,
        num_workers=0, pin_memory=False
    )

    # 模型初始化
    model = YOLO12_1D_Classifier(config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = GradScaler()

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss, train_acc, train_total = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Train")
        optimizer.zero_grad()

        for batch_idx, (data, labels) in enumerate(pbar):
            # 显存清理
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # 混合精度训练
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels.to(device)) / config.ACCUMULATION_STEPS

            # 反向传播
            scaler.scale(loss).backward()
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 统计
            train_loss += loss.item() * config.ACCUMULATION_STEPS * data.size(0)
            train_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
            train_total += data.size(0)
            mem_used = torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({
                'Loss': f'{loss.item()*config.ACCUMULATION_STEPS:.4f}',
                'Acc': f'{train_acc/train_total:.4f}',
                'Mem': f'{mem_used:.1f}GB'
            })

        # 验证阶段
        model.eval()
        val_loss, val_acc, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{config.EPOCHS}] Val"):
                outputs = model(data)
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item() * data.size(0)
                val_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
                val_total += data.size(0)

        # 结果统计
        train_loss_avg = train_loss / train_total
        train_acc_avg = train_acc / train_total
        val_loss_avg = val_loss / val_total
        val_acc_avg = val_acc / val_total
        scheduler.step()

        # 打印结果
        print(f"\n📊 Epoch {epoch+1} 结果：")
        print(f"  - 训练损失：{train_loss_avg:.4f} | 训练准确率：{train_acc_avg:.4f}")
        print(f"  - 验证损失：{val_loss_avg:.4f} | 验证准确率：{val_acc_avg:.4f}")

        # 保存最优模型
        if val_acc_avg > best_val_acc:
            best_val_acc = val_acc_avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, os.path.join(config.DATASET_OUTPUT_DIR, "yolo12_1d_best.pth"))
            print(f"✅ 保存最优模型（验证准确率：{best_val_acc:.4f}）")

    # 测试阶段
    print("\n📌 测试阶段（加载最优模型）")
    checkpoint = torch.load(os.path.join(config.DATASET_OUTPUT_DIR, "yolo12_1d_best.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_acc, test_total = 0.0, 0
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing"):
            outputs = model(data)
            test_acc += (outputs.argmax(1) == labels.to(device)).sum().item()
            test_total += data.size(0)
    test_acc_avg = test_acc / test_total

    # 最终结果
    print("\n" + "=" * 80)
    print("🎉 训练完成！最终结果：")
    print(f"  - 最优验证准确率：{best_val_acc:.4f}")
    print(f"  - 测试集准确率：{test_acc_avg:.4f}")
    print(f"  - 模型保存至：{os.path.join(config.DATASET_OUTPUT_DIR, 'yolo12_1d_best.pth')}")
    print("=" * 80)

# ===================== 运行入口 =====================
if __name__ == "__main__":
    train_model()