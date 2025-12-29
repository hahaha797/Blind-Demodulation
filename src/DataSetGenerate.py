import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# ===================== 配置（仅修改此处） =====================
class Config:
    DATA_DIR = "../../DataSet"  # 原始.bin/.wav文件目录
    DATASET_OUTPUT_DIR = "./modulation_dataset"  # 数据集输出目录
    SAMPLE_LENGTH = 4096  # 单样本IQ长度

    # 数据集划分
    TEST_SIZE = 0.1
    VAL_SIZE = 0.111
    RANDOM_STATE = 42

    # 快速测试采样（跑通后注释）
    # TRAIN_SAMPLE_LIMIT = 10000
    # VAL_SAMPLE_LIMIT = 1000
    # TEST_SAMPLE_LIMIT = 1000


config = Config()
os.makedirs(config.DATASET_OUTPUT_DIR, exist_ok=True)


# ===================== 核心函数：构造数据集 =====================
def get_file_iq_info(file_path):
    """解析单个文件的IQ信息（复用你的逻辑）"""
    try:
        if file_path.endswith('.bin'):
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.int16)
            total_iq = len(data) // 2
            file_type = 'bin'
            frame_size = 131072
            total_frames = total_iq // frame_size
            valid_iq = total_frames * frame_size
            num_samples_per_file = sum([frame_size - config.SAMPLE_LENGTH + 1 for _ in range(total_frames)])

        elif file_path.endswith('.wav'):
            with open(file_path, 'rb') as f:
                f.seek(1068)
                data = np.fromfile(f, dtype=np.int16)
            total_iq = len(data) // 2
            file_type = 'wav'
            valid_iq = total_iq
            num_samples_per_file = total_iq - config.SAMPLE_LENGTH + 1 if total_iq >= config.SAMPLE_LENGTH else 0

        else:
            return None

        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        modulation = name_without_ext.split('_')[0]
        sample_rate_match = re.search(r'(\d+\.?\d*)\s*([kM]SPS)', name_without_ext)
        if sample_rate_match:
            num = float(sample_rate_match.group(1))
            unit = sample_rate_match.group(2)
            sample_rate = num * 1e3 if unit == 'kSPS' else num * 1e6
        else:
            sample_rate = None

        return {
            'file_path': file_path,
            'filename': filename,
            'file_type': file_type,
            'modulation': modulation,
            'sample_rate_hz': sample_rate,
            'total_iq_pairs': total_iq,
            'valid_iq_pairs': valid_iq,
            'num_samples': num_samples_per_file,
            'sample_length': config.SAMPLE_LENGTH,
            'step': 1
        }
    except Exception as e:
        print(f"⚠️  处理文件失败：{file_path} -> {str(e)}")
        return None


def construct_dataset():
    """主函数：生成元数据 + npy数据集"""
    print("=" * 80)
    print("🚀 开始构造数据集（仅运行一次）")
    print("=" * 80)

    # 1. 生成全局样本映射
    all_file_metadata = []
    global_sample_counter = 0
    global_sample_mapping = []

    for filename in os.listdir(config.DATA_DIR):
        file_path = os.path.join(config.DATA_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        file_info = get_file_iq_info(file_path)
        if not file_info or file_info['num_samples'] == 0:
            continue

        all_file_metadata.append(file_info)

        # 生成样本映射
        if file_info['file_type'] == 'bin':
            frame_size = 131072
            frame_start = 0
            for frame_idx in range(file_info['valid_iq_pairs'] // frame_size):
                frame_samples = frame_size - config.SAMPLE_LENGTH + 1
                for frame_inner_start in range(frame_samples):
                    global_start = frame_start + frame_inner_start
                    global_sample_mapping.append({
                        'global_idx': global_sample_counter,
                        'file_path': file_path,
                        'start_iq_idx': global_start,
                        'modulation': file_info['modulation']
                    })
                    global_sample_counter += 1
                frame_start += frame_size
        else:
            for start_iq_idx in range(file_info['num_samples']):
                global_sample_mapping.append({
                    'global_idx': global_sample_counter,
                    'file_path': file_path,
                    'start_iq_idx': start_iq_idx,
                    'modulation': file_info['modulation']
                })
                global_sample_counter += 1

        print(f"📄 处理完成：{filename} → {file_info['num_samples']}个样本")

    # 保存元数据
    pd.DataFrame(all_file_metadata).to_csv(
        os.path.join(config.DATASET_OUTPUT_DIR, "file_metadata.csv"),
        index=False, encoding='utf-8'
    )
    pd.DataFrame(global_sample_mapping).to_csv(
        os.path.join(config.DATASET_OUTPUT_DIR, "global_sample_mapping.csv"),
        index=False, encoding='utf-8'
    )

    # 生成标签映射
    all_modulations = sorted(list(set([f['modulation'] for f in all_file_metadata])))
    label_encoder_mapping = {mod: idx for idx, mod in enumerate(all_modulations)}
    with open(os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json"), 'w') as f:
        json.dump({
            'label_to_idx': label_encoder_mapping,
            'idx_to_label': {v: k for k, v in label_encoder_mapping.items()},
            'total_samples': global_sample_counter
        }, f, indent=4)

    # 2. 生成npy数据集
    print("\n📌 开始生成npy数据集（训练全程用这个）")
    mapping_df = pd.read_csv(os.path.join(config.DATASET_OUTPUT_DIR, "global_sample_mapping.csv"))
    label_mapping = json.load(open(os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json"), 'r'))
    label_to_idx = label_mapping['label_to_idx']
    total_samples = label_mapping['total_samples']

    # 分层划分
    X = np.arange(total_samples)
    y = np.zeros(total_samples)
    X_train_val, X_test, _, _ = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    X_train, X_val, _, _ = train_test_split(
        X_train_val, y[:len(X_train_val)], test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
    )

    # 采样限制
    if hasattr(config, 'TRAIN_SAMPLE_LIMIT'):
        X_train = X_train[:config.TRAIN_SAMPLE_LIMIT]
    if hasattr(config, 'VAL_SAMPLE_LIMIT'):
        X_val = X_val[:config.VAL_SAMPLE_LIMIT]
    if hasattr(config, 'TEST_SAMPLE_LIMIT'):
        X_test = X_test[:config.TEST_SAMPLE_LIMIT]

    # 读取单个样本
    def read_sample(global_idx):
        row = mapping_df.iloc[global_idx]
        file_path = row['file_path'].replace('/', '\\')
        if not os.path.exists(file_path):
            file_path = os.path.join(config.DATA_DIR, os.path.basename(file_path))

        start_idx = int(row['start_iq_idx'])
        modulation = row['modulation']

        # 读取IQ数据
        if file_path.endswith('.bin'):
            with open(file_path, 'rb') as f:
                f.seek(start_idx * 4)
                data = np.fromfile(f, dtype=np.int16, count=config.SAMPLE_LENGTH * 2)
        elif file_path.endswith('.wav'):
            with open(file_path, 'rb') as f:
                f.seek(1068 + start_idx * 4)
                data = np.fromfile(f, dtype=np.int16, count=config.SAMPLE_LENGTH * 2)
        else:
            return np.zeros((config.SAMPLE_LENGTH, 2), dtype=np.float32), 0

        # 处理数据
        if len(data) < config.SAMPLE_LENGTH * 2:
            sample = np.zeros((config.SAMPLE_LENGTH, 2), dtype=np.int16)
            valid_len = len(data) // 2
            sample[:valid_len] = data.reshape(-1, 2)
        else:
            sample = data.reshape(-1, 2)
        sample_norm = sample.astype(np.float32) / 32767.0
        label = label_to_idx.get(modulation, 0)
        return sample_norm, label

    # 生成训练集
    print("📌 生成训练集...")
    train_data, train_labels = [], []
    for idx in tqdm(X_train, desc="Train"):
        data, label = read_sample(idx)
        train_data.append(data)
        train_labels.append(label)
    train_data = np.array(train_data).transpose(0, 2, 1)
    np.save(os.path.join(config.DATASET_OUTPUT_DIR, "train_data.npy"), train_data)
    np.save(os.path.join(config.DATASET_OUTPUT_DIR, "train_labels.npy"), np.array(train_labels))

    # 生成验证集
    print("📌 生成验证集...")
    val_data, val_labels = [], []
    for idx in tqdm(X_val, desc="Val"):
        data, label = read_sample(idx)
        val_data.append(data)
        val_labels.append(label)
    val_data = np.array(val_data).transpose(0, 2, 1)
    np.save(os.path.join(config.DATASET_OUTPUT_DIR, "val_data.npy"), val_data)
    np.save(os.path.join(config.DATASET_OUTPUT_DIR, "val_labels.npy"), np.array(val_labels))

    # 生成测试集
    print("📌 生成测试集...")
    test_data, test_labels = [], []
    for idx in tqdm(X_test, desc="Test"):
        data, label = read_sample(idx)
        test_data.append(data)
        test_labels.append(label)
    test_data = np.array(test_data).transpose(0, 2, 1)
    np.save(os.path.join(config.DATASET_OUTPUT_DIR, "test_data.npy"), test_data)
    np.save(os.path.join(config.DATASET_OUTPUT_DIR, "test_labels.npy"), np.array(test_labels))

    # 完成提示
    print("\n" + "=" * 80)
    print("🎉 数据集构造完成！生成的文件：")
    file_list = [
        "file_metadata.csv", "global_sample_mapping.csv", "label_mapping.json",
        "train_data.npy", "train_labels.npy", "val_data.npy", "val_labels.npy", "test_data.npy", "test_labels.npy"
    ]
    for f in file_list:
        print(f"  - {os.path.join(config.DATASET_OUTPUT_DIR, f)}")
    print("✅ 接下来运行trainer.py训练，全程不碰原始文件！")
    print("=" * 80)


# ===================== 运行入口 =====================
if __name__ == "__main__":
    construct_dataset()