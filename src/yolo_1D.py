import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -------------------------- 一维YOLO12核心模块 --------------------------
class Conv1d(nn.Module):
    """一维卷积模块（替代YOLO的Conv2d）：Conv1d + BN + SiLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # 保持维度不变
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()  # YOLO默认激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck1d(nn.Module):
    """一维瓶颈模块（C2f的核心）"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv1d(c1, c_, 1, 1)
        self.cv2 = Conv1d(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f1d(nn.Module):
    """一维C2f模块（YOLO12骨干核心）"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv1d(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv1d((2 + n) * self.c_, c2, 1)  # 拼接后卷积
        self.m = nn.ModuleList(Bottleneck1d(self.c_, self.c_, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # 拆分为2个分支
        y.extend(m(y[-1]) for m in self.m)  # 残差分支
        return self.cv2(torch.cat(y, 1))  # 拼接后卷积


class SPPF1d(nn.Module):
    """一维SPPF模块（空间金字塔池化，替换二维SPPF）"""

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


# -------------------------- 一维YOLO12分类模型 --------------------------
class YOLO12_1D_Classifier(nn.Module):
    """
    一维YOLO12调制信号分类模型
    输入：(batch, 2, 4096) → 2通道（I/Q），4096长度IQ序列
    输出：(batch, num_classes) → 调制类型分类概率
    """

    def __init__(self, num_classes=20, in_channels=2, base_channels=32):
        super().__init__()
        # 1. 骨干网络（YOLO12 Backbone，全一维化）
        self.backbone = nn.Sequential(
            # 输入：2→32，长度4096→2048（stride=2）
            Conv1d(in_channels, base_channels, 6, 2, 2),
            # 下采样1：32→64，长度2048→1024
            C2f1d(base_channels, base_channels * 2, n=1, shortcut=True),
            Conv1d(base_channels * 2, base_channels * 2, 3, 2, 1),
            # 下采样2：64→128，长度1024→512
            C2f1d(base_channels * 2, base_channels * 4, n=2, shortcut=True),
            Conv1d(base_channels * 4, base_channels * 4, 3, 2, 1),
            # 下采样3：128→256，长度512→256
            C2f1d(base_channels * 4, base_channels * 8, n=2, shortcut=True),
            Conv1d(base_channels * 8, base_channels * 8, 3, 2, 1),
            # 下采样4：256→512，长度256→128
            C2f1d(base_channels * 8, base_channels * 16, n=3, shortcut=True),
            Conv1d(base_channels * 16, base_channels * 16, 3, 2, 1),
            # SPPF增强特征
            C2f1d(base_channels * 16, base_channels * 16, n=3, shortcut=True),
            SPPF1d(base_channels * 16, base_channels * 16, k=5),
        )

        # 2. 分类头（替换YOLO检测头）
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化：(512,128)→(512,1)
            nn.Flatten(),  # 展平：(512,1)→512
            nn.Linear(base_channels * 16, base_channels * 8),  # 512→256
            nn.Dropout(0.2),  # 防止过拟合
            nn.SiLU(),
            nn.Linear(base_channels * 8, num_classes)  # 256→分类数
        )

    def forward(self, x):
        # x: (batch, 2, 4096)
        x = self.backbone(x)  # (batch, 512, 128)
        x = self.class_head(x)  # (batch, num_classes)
        return x


# -------------------------- 数据集加载（复用之前的动态加载类） --------------------------
class DynamicSlidingWindowDataset(nn.Module):
    """复用之前的动态滑动窗口数据集类（略，完整代码见上一轮回复）"""

    def __init__(self, split='train', test_size=0.2, val_size=0.125, random_state=42):
        import pandas as pd
        import json
        from sklearn.model_selection import train_test_split
        METADATA_DIR = "./modulation_metadata"

        # 加载元数据
        self.sample_mapping = pd.read_csv(f"{METADATA_DIR}/global_sample_mapping.csv")
        self.label_mapping = json.load(open(f"{METADATA_DIR}/label_mapping.json", 'r'))
        self.total_samples = self.label_mapping['total_samples']
        self.label_to_idx = self.label_mapping['label_to_idx']
        self.num_classes = len(self.label_to_idx)

        # 分层划分数据集
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
        print(f"✅ {split}集初始化完成：{len(self.selected_idxs)}个样本")

    def __len__(self):
        return len(self.selected_idxs)

    def _read_iq_data(self, file_path, start_idx, length):
        import numpy as np
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
        import numpy as np
        global_idx = self.idx_map[idx]
        sample_info = self.sample_mapping[self.sample_mapping['global_idx'] == global_idx].iloc[0]
        file_path = sample_info['file_path']
        start_iq_idx = int(sample_info['start_iq_idx'])
        label_idx = self.label_to_idx[sample_info['modulation']]

        iq_data = self._read_iq_data(file_path, start_iq_idx, 4096)
        sample_tensor = torch.from_numpy(iq_data).permute(1, 0).float()  # (2, 4096)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return sample_tensor, label_tensor


# -------------------------- 训练脚本 --------------------------
def train_yolo12_1d():
    # 1. 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    batch_size = 32
    lr = 1e-3
    weight_decay = 5e-4
    num_classes = 20  # 根据实际调制类型数调整

    # 2. 加载数据集
    train_dataset = DynamicSlidingWindowDataset(split='train')
    val_dataset = DynamicSlidingWindowDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. 初始化模型
    model = YOLO12_1D_Classifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # 分类损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. 训练循环
    best_val_acc = 0.0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算指标
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total

        # 更新学习率
        scheduler.step()

        # 打印epoch结果
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, f"yolo12_1d_best.pth")
            print(f"✅ 保存最优模型（Val Acc: {best_val_acc:.4f}）")

    # 5. 测试集评估
    test_dataset = DynamicSlidingWindowDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.load_state_dict(torch.load("yolo12_1d_best.pth")['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print(f"\n最终测试集准确率：{test_acc:.4f}")


# -------------------------- 推理脚本（单样本预测） --------------------------
def predict_single_sample(model_path, sample_tensor):
    """
    单样本预测
    model_path: 模型权重路径
    sample_tensor: (2, 4096)的IQ张量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 20
    label_mapping = json.load(open("./modulation_metadata/label_mapping.json", 'r'))
    idx_to_label = label_mapping['idx_to_label']

    # 加载模型
    model = YOLO12_1D_Classifier(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 预测
    with torch.no_grad():
        sample_tensor = sample_tensor.unsqueeze(0).to(device)  # (1, 2, 4096)
        outputs = model(sample_tensor)
        prob = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        pred_modulation = idx_to_label[str(pred_idx)]
        pred_conf = prob[0][pred_idx].item()

    return {
        'pred_modulation': pred_modulation,
        'confidence': pred_conf,
        'all_probs': prob.cpu().numpy()[0].tolist()
    }


# -------------------------- 运行入口 --------------------------
if __name__ == "__main__":
    # 训练模型
    train_yolo12_1d()

    # 单样本预测示例
    # test_dataset = DynamicSlidingWindowDataset(split='test')
    # sample, _ = test_dataset[0]
    # result = predict_single_sample("yolo12_1d_best.pth", sample)
    # print(f"预测结果：{result['pred_modulation']}，置信度：{result['confidence']:.4f}")