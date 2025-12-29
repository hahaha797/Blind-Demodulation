import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 强制设置GPU环境 --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# -------------------------- 一维YOLO12核心模块 --------------------------
class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
        self.to(torch.device('cuda'))

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck1d(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv1d(c1, c_, 1, 1)
        self.cv2 = Conv1d(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f1d(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv1d(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv1d((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck1d(self.c_, self.c_, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF1d(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv1d(c1, c_, 1, 1)
        self.cv2 = Conv1d(c_ * 4, c2, 1, 1)
        self.k1 = nn.MaxPool1d(kernel_size=k, stride=1, padding=k // 2)
        self.k2 = nn.MaxPool1d(kernel_size=k * 2, stride=1, padding=k)
        self.k3 = nn.MaxPool1d(kernel_size=k * 4, stride=1, padding=2 * k)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.k1(x)
        y2 = self.k2(x)
        y3 = self.k3(x)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# -------------------------- 一维YOLO12分类模型（GPU版） --------------------------
class YOLO12_1D_Classifier(nn.Module):
    def __init__(self, num_classes=20, in_channels=2, base_channels=32):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert self.device.type == 'cuda', "未检测到GPU！请检查CUDA安装"

        self.backbone = nn.Sequential(
            Conv1d(in_channels, base_channels, 6, 2, 2),
            C2f1d(base_channels, base_channels * 2, n=1, shortcut=True),
            Conv1d(base_channels * 2, base_channels * 2, 3, 2, 1),
            C2f1d(base_channels * 2, base_channels * 4, n=2, shortcut=True),
            Conv1d(base_channels * 4, base_channels * 4, 3, 2, 1),
            C2f1d(base_channels * 4, base_channels * 8, n=2, shortcut=True),
            Conv1d(base_channels * 8, base_channels * 8, 3, 2, 1),
            C2f1d(base_channels * 8, base_channels * 16, n=3, shortcut=True),
            Conv1d(base_channels * 16, base_channels * 16, 3, 2, 1),
            C2f1d(base_channels * 16, base_channels * 16, n=3, shortcut=True),
            SPPF1d(base_channels * 16, base_channels * 16, k=5),
        ).to(self.device)

        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1).to(self.device),
            nn.Flatten().to(self.device),
            nn.Linear(base_channels * 16, base_channels * 8).to(self.device),
            nn.Dropout(0.2).to(self.device),
            nn.SiLU().to(self.device),
            nn.Linear(base_channels * 8, num_classes).to(self.device)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.backbone(x)
        x = self.class_head(x)
        return x


# -------------------------- 动态滑动窗口数据集 --------------------------
class DynamicSlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', test_size=0.2, val_size=0.125, random_state=42):
        self.METADATA_DIR = "./modulation_metadata"
        self.SAMPLE_LENGTH = 4096

        self.sample_mapping = pd.read_csv(f"{self.METADATA_DIR}/global_sample_mapping.csv")
        self.label_mapping = json.load(open(f"{self.METADATA_DIR}/label_mapping.json", 'r'))
        self.total_samples = self.label_mapping['total_samples']
        self.label_to_idx = self.label_mapping['label_to_idx']
        self.num_classes = len(self.label_to_idx)

        self.sample_mapping['label_idx'] = self.sample_mapping['modulation'].map(self.label_to_idx)
        X = self.sample_mapping['global_idx'].values
        y = self.sample_mapping['label_idx'].values

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )

        if split == 'train':
            self.selected_idxs = X_train
        elif split == 'val':
            self.selected_idxs = X_val
        elif split == 'test':
            self.selected_idxs = X_test
        else:
            raise ValueError(f"split must be 'train'/'val'/'test'")

        self.idx_map = {ds_idx: global_idx for ds_idx, global_idx in enumerate(self.selected_idxs)}
        print(f"✅ {split}集初始化完成：{len(self.selected_idxs)}个样本（GPU加速加载）")

    def __len__(self):
        return len(self.selected_idxs)

    def _read_iq_data(self, file_path, start_idx, length):
        try:
            if file_path.endswith('.bin'):
                with open(file_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.int16)
                iq_data = data.reshape(-1, 2)
            elif file_path.endswith('.wav'):
                with open(file_path, 'rb') as f:
                    f.seek(1068)
                    data = np.fromfile(f, dtype=np.int16)
                iq_data = data.reshape(-1, 2)
            else:
                raise ValueError(f"不支持的文件类型：{file_path}")

            end_idx = start_idx + length
            if end_idx > len(iq_data):
                sample = np.zeros((length, 2), dtype=np.int16)
                valid_len = len(iq_data) - start_idx
                sample[:valid_len] = iq_data[start_idx:]
            else:
                sample = iq_data[start_idx:end_idx]

            sample_norm = sample.astype(np.float32) / 32767.0
            return sample_norm
        except Exception as e:
            print(f"⚠️  读取失败：{file_path} → {str(e)}")
            return np.zeros((length, 2), dtype=np.float32)

    def __getitem__(self, idx):
        global_idx = self.idx_map[idx]
        sample_info = self.sample_mapping[self.sample_mapping['global_idx'] == global_idx].iloc[0]
        file_path = sample_info['file_path']
        start_iq_idx = int(sample_info['start_iq_idx'])
        label_idx = self.label_to_idx[sample_info['modulation']]

        iq_data = self._read_iq_data(file_path, start_iq_idx, self.SAMPLE_LENGTH)
        sample_tensor = torch.from_numpy(iq_data).permute(1, 0).float().contiguous()
        label_tensor = torch.tensor(label_idx, dtype=torch.long).contiguous()

        return sample_tensor, label_tensor


# -------------------------- GPU训练核心函数 --------------------------
def train_yolo12_1d_gpu():
    device = torch.device('cuda')
    print(f"📌 使用GPU训练：{torch.cuda.get_device_name(0)}")
    print(f"📌 GPU显存总量：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GiB")

    epochs = 50
    batch_size = 64
    lr = 1e-3
    weight_decay = 5e-4
    num_classes = 20
    accumulation_steps = 2

    train_dataset = DynamicSlidingWindowDataset(split='train')
    val_dataset = DynamicSlidingWindowDataset(split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        prefetch_factor=2
    )

    model = YOLO12_1D_Classifier(num_classes=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"📌 启用多GPU训练：{torch.cuda.device_count()}块GPU")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Train")

        optimizer.zero_grad()
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps * data.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            mem_used = torch.cuda.memory_allocated(0) / 1024 ** 3
            pbar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'Acc': f'{train_correct / train_total:.4f}',
                'Mem': f'{mem_used:.2f}GiB'
            })

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Val"):
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        print(f"\n📊 Epoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")
        print(
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | GPU Mem Used: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f}GiB")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'scaler_state_dict': scaler.state_dict()
            }, f"yolo12_1d_gpu_best.pth")
            print(f"✅ 保存最优GPU模型（Val Acc: {best_val_acc:.4f}）")

        torch.cuda.empty_cache()

    print("\n🔍 测试集最终评估")
    test_dataset = DynamicSlidingWindowDataset(split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    model.load_state_dict(torch.load("yolo12_1d_gpu_best.pth")['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Testing"):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print(f"\n🏆 最终测试集准确率：{test_acc:.4f}")
    print(f"🏆 最优验证集准确率：{best_val_acc:.4f}")


# -------------------------- 修复后的GPU推理函数（核心修改） --------------------------
def predict_gpu(model_path, sample_tensor):
    """
    修复推理时间计算：正确使用torch.cuda.Event记录GPU推理耗时
    """
    device = torch.device('cuda')
    num_classes = 20
    label_mapping = json.load(open("./modulation_metadata/label_mapping.json", 'r'))
    idx_to_label = label_mapping['idx_to_label']

    # 1. 加载模型（GPU）
    model = YOLO12_1D_Classifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 初始化GPU时间事件（关键修复）
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 3. GPU推理（带精确计时）
    with torch.no_grad(), autocast():
        sample_tensor = sample_tensor.unsqueeze(0).to(device, non_blocking=True)

        # 预热（避免首次推理耗时偏高）
        for _ in range(5):
            _ = model(sample_tensor)

        # 记录开始时间
        torch.cuda.synchronize()  # 等待之前的GPU操作完成
        start_event.record()

        # 执行推理
        outputs = model(sample_tensor)

        # 记录结束时间并同步
        end_event.record()
        torch.cuda.synchronize()  # 等待推理完成

        # 计算耗时（毫秒）
        inference_time = start_event.elapsed_time(end_event)

        # 计算预测结果
        prob = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        pred_modulation = idx_to_label[str(pred_idx)]
        pred_conf = prob[0][pred_idx].item()

    return {
        'pred_modulation': pred_modulation,
        'confidence': round(pred_conf, 4),
        'inference_time': f"{inference_time:.2f}ms"  # 修复后的耗时输出
    }


# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到GPU！请安装CUDA并配置PyTorch-GPU版本")

    # 开始GPU训练
    train_yolo12_1d_gpu()

    # 单样本GPU推理示例
    # test_dataset = DynamicSlidingWindowDataset(split='test')
    # sample, _ = test_dataset[0]
    # result = predict_gpu("yolo12_1d_gpu_best.pth", sample)
    # print(f"\n📝 推理结果：{result['pred_modulation']}（置信度：{result['confidence']}）")
    # print(f"📝 GPU推理耗时：{result['inference_time']}")