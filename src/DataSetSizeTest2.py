import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader

# -------------------------- 配置参数 --------------------------
METADATA_DIR = "./modulation_metadata"
SAMPLE_LENGTH = 4096  # 4096对IQ数据/样本


# -------------------------- 动态加载Dataset类 --------------------------
class DynamicSlidingWindowDataset(Dataset):
    """
    动态滑动窗口加载数据集（步长1）
    核心：访问样本时才从原始文件提取对应位置的4096对IQ数据，不预先保存所有样本
    """

    def __init__(self, split='train', test_size=0.2, val_size=0.125, random_state=42):
        # 加载元数据
        self.sample_mapping = pd.read_csv(f"{METADATA_DIR}/global_sample_mapping.csv")
        self.label_mapping = json.load(open(f"{METADATA_DIR}/label_mapping.json", 'r'))
        self.total_samples = self.label_mapping['total_samples']
        self.label_to_idx = self.label_mapping['label_to_idx']

        # 分层划分训练/验证/测试集（保证调制类型分布均匀）
        from sklearn.model_selection import train_test_split

        # 按调制类型分组，分层划分
        self.sample_mapping['label_idx'] = self.sample_mapping['modulation'].map(self.label_to_idx)
        X = self.sample_mapping['global_idx'].values
        y = self.sample_mapping['label_idx'].values

        # 第一步：划分训练+验证集 和 测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 第二步：划分训练集 和 验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )

        # 确定当前数据集的样本索引
        if split == 'train':
            self.selected_idxs = X_train
        elif split == 'val':
            self.selected_idxs = X_val
        elif split == 'test':
            self.selected_idxs = X_test
        else:
            raise ValueError(f"split must be 'train'/'val'/'test', got {split}")

        # 构建索引映射：Dataset索引 → 全局样本索引
        self.idx_map = {ds_idx: global_idx for ds_idx, global_idx in enumerate(self.selected_idxs)}
        print(f"✅ {split}集初始化完成：{len(self.selected_idxs)}个样本")

    def __len__(self):
        return len(self.selected_idxs)

    def _read_iq_data(self, file_path, start_idx, length):
        """从指定文件的指定起始位置读取length对IQ数据"""
        try:
            if file_path.endswith('.bin'):
                with open(file_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.int16)
                iq_data = data.reshape(-1, 2)

            elif file_path.endswith('.wav'):
                with open(file_path, 'rb') as f:
                    f.seek(1068)  # 跳过头部
                    data = np.fromfile(f, dtype=np.int16)
                iq_data = data.reshape(-1, 2)

            else:
                raise ValueError(f"不支持的文件类型：{file_path}")

            # 提取指定位置的IQ数据
            end_idx = start_idx + length
            if end_idx > len(iq_data):
                # 边界处理：不足时补0（实际不会触发，元数据已过滤）
                sample = np.zeros((length, 2), dtype=np.int16)
                valid_len = len(iq_data) - start_idx
                sample[:valid_len] = iq_data[start_idx:]
            else:
                sample = iq_data[start_idx:end_idx]

            # 归一化：short(-32768~32767) → float32(-1.0~1.0)
            sample_norm = sample.astype(np.float32) / 32767.0
            return sample_norm

        except Exception as e:
            print(f"⚠️  读取IQ数据失败：{file_path} start={start_idx} → {str(e)}")
            return np.zeros((length, 2), dtype=np.float32)

    def __getitem__(self, idx):
        # 1. 获取全局样本索引
        global_idx = self.idx_map[idx]

        # 2. 查找该样本对应的文件和起始位置
        sample_info = self.sample_mapping[self.sample_mapping['global_idx'] == global_idx].iloc[0]
        file_path = sample_info['file_path']
        start_iq_idx = int(sample_info['start_iq_idx'])
        modulation = sample_info['modulation']
        label_idx = self.label_to_idx[modulation]

        # 3. 动态读取IQ数据（步长1的4096对IQ）
        iq_data = self._read_iq_data(file_path, start_iq_idx, SAMPLE_LENGTH)

        # 4. 转换为PyTorch张量（C, L）→ 通道在前
        sample_tensor = torch.from_numpy(iq_data).permute(1, 0).float()
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return sample_tensor, label_tensor


# -------------------------- 测试加载（查看形状） --------------------------
def dynamic_loading():
    print("=" * 70)
    print("📊 测试动态滑动窗口加载（步长1）")
    print("=" * 70)

    # 初始化训练集
    train_dataset = DynamicSlidingWindowDataset(split='train')
    val_dataset = DynamicSlidingWindowDataset(split='val')
    test_dataset = DynamicSlidingWindowDataset(split='test')

    # 查看单个样本形状
    sample, label = train_dataset[0]
    print(f"🔍 单个样本形状：{sample.shape} → (通道数=2, 序列长度=4096)")
    print(f"🔍 单个样本标签：{label.item()} → {train_dataset.label_mapping['idx_to_label'][str(label.item())]}")

    # 批量加载测试
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    batch_data, batch_labels = next(iter(train_loader))
    print(f"\n📦 批量加载形状：")
    print(f"  - 批量特征：{batch_data.shape} → (batch_size=16, 通道数=2, 序列长度=4096)")
    print(f"  - 批量标签：{batch_labels.shape} → (batch_size=16,)")
    print(f"  - 批量标签示例：{batch_labels[:5].numpy()}")

    # 数据集大小统计
    print(f"\n📈 数据集大小：")
    print(f"  - 训练集：{len(train_dataset)} 样本")
    print(f"  - 验证集：{len(val_dataset)} 样本")
    print(f"  - 测试集：{len(test_dataset)} 样本")

    print("\n" + "=" * 70)
    print("✅ 动态滑动窗口加载测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    dynamic_loading()