import torch
import numpy as np
import json
import torch.serialization
from torch.utils.data import DataLoader
from numpy import ndarray
from numpy._core.multiarray import _reconstruct

# -------------------------- 配置路径 --------------------------
DATASET_DIR = "./pytorch_modulation_dataset_4096"
TRAIN_PATH = f"{DATASET_DIR}/train_dataset.pt"
VAL_PATH = f"{DATASET_DIR}/val_dataset.pt"
TEST_PATH = f"{DATASET_DIR}/test_dataset.pt"
LABEL_MAPPING_PATH = f"{DATASET_DIR}/label_mapping.json"

# -------------------------- 复刻生成程序的Dataset类 --------------------------
class ModulationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_tensor = torch.from_numpy(self.samples[idx]).permute(1, 0).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample_tensor, label_tensor

# -------------------------- 解决安全加载限制 --------------------------
torch.serialization.add_safe_globals([
    ModulationDataset,
    ndarray,
    _reconstruct
])

# -------------------------- 加载并显示标签详情 --------------------------
def load_and_check_data_with_labels():
    print("=" * 70)
    print("📊 调制信号数据集形状+标签查看工具")
    print("=" * 70)
    print(f"🔍 数据集目录：{DATASET_DIR}")
    print("=" * 70)

    # 1. 加载数据集
    try:
        train_dataset = torch.load(TRAIN_PATH, weights_only=False)
        val_dataset = torch.load(VAL_PATH, weights_only=False)
        test_dataset = torch.load(TEST_PATH, weights_only=False)
        print("✅ 数据集加载成功！")
    except Exception as e:
        print(f"❌ 数据集加载失败：{str(e)}")
        return

    # 2. 加载标签映射表（关键：获取整数标签→调制类型的映射）
    try:
        with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        idx_to_label = label_mapping['idx_to_label']  # 整数→调制类型（字符串）
        print(f"✅ 标签映射表加载成功！共{len(idx_to_label)}种调制类型")
    except Exception as e:
        print(f"⚠️  标签映射表加载失败：{str(e)}")
        idx_to_label = None

    # 3. 提取数据形状+标签信息
    def get_data_info(dataset, name):
        total_samples = len(dataset)
        sample, label = dataset[0]
        # 取前5个样本的标签示例
        sample_labels = [dataset[i][1].item() for i in range(min(5, total_samples))]
        return {
            "name": name,
            "total_samples": total_samples,
            "sample_shape": sample.shape,
            "label_shape": label.shape,
            "sample_dtype": sample.dtype,
            "label_dtype": label.dtype,
            "sample_labels": sample_labels  # 前5个样本的整数标签
        }

    # 4. 输出详细信息（含标签）
    datasets_info = [
        get_data_info(train_dataset, "训练集"),
        get_data_info(val_dataset, "验证集"),
        get_data_info(test_dataset, "测试集")
    ]

    for info in datasets_info:
        print(f"\n📈 {info['name']} 信息：")
        print(f"  - 总样本数：{info['total_samples']}")
        print(f"  - 单个样本形状：{info['sample_shape']}（通道数={info['sample_shape'][0]}, 序列长度={info['sample_shape'][1]}）")
        print(f"  - 样本数据类型：{info['sample_dtype']}")
        print(f"  - 标签数据类型：{info['label_dtype']}（整数编码）")
        print(f"  - 前5个样本的整数标签：{info['sample_labels']}")
        # 显示标签对应的调制类型名称（若映射表加载成功）
        if idx_to_label:
            sample_mod_types = [idx_to_label[str(idx)] for idx in info['sample_labels']]
            print(f"  - 对应调制类型：{sample_mod_types}")

    # 5. 批量标签示例
    print(f"\n" + "-" * 50)
    print("📦 批量数据标签示例（batch_size=16）：")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    batch_data, batch_labels = next(iter(train_loader))
    print(f"  - 批量特征形状：{batch_data.shape}")
    print(f"  - 批量标签形状：{batch_labels.shape}（16个样本的标签）")
    print(f"  - 批量标签数值：{batch_labels.numpy()}")
    if idx_to_label:
        batch_mod_types = [idx_to_label[str(idx)] for idx in batch_labels.numpy()[:5]]
        print(f"  - 前5个批量标签对应类型：{batch_mod_types}")

    # 6. numpy格式标签查看（可选）
    print(f"\n" + "-" * 50)
    print("📌 numpy格式标签查看（若已保存）：")
    try:
        y_train_np = np.load(f"{DATASET_DIR}/y_train.npy")
        print(f"  - numpy训练集标签形状：{y_train_np.shape}")
        print(f"  - 前10个numpy标签：{y_train_np[:10]}")
    except FileNotFoundError:
        print("  - 未找到numpy格式标签文件")

    print("\n" + "=" * 70)
    print("✅ 验证完成：数据集包含完整标签！")
    print("📝 标签说明：整数标签 → 对应调制类型（见label_mapping.json）")
    print("=" * 70)

if __name__ == "__main__":
    load_and_check_data_with_labels()