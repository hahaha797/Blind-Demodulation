import os
import re
import json
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import shutil

warnings.filterwarnings('ignore')


# ===================== 配置（优化版） =====================
class Config:
    # 数据路径配置
    DATA_DIR = "../../DataSet"  # 原始.bin/.wav文件目录
    DATASET_OUTPUT_DIR = "./modulation_dataset_50overlap"  # 数据集输出目录

    SAMPLE_LENGTH = 4096  # 单样本IQ长度

    # === 核心优化 1：步长设置 (50% 重叠) ===
    # 步长 = 2048，即每次移动半个窗口，既增加了数据量又避免了过度膨胀
    STRIDE = 2048

    # === 核心优化 2：数据精度 ===
    # 使用 float16 替代 float32，体积减半，显存占用减半
    DTYPE = np.float16

    # 流式分块配置
    CHUNK_SIZE = 100000  # 每块样本数

    # 数据集划分
    TEST_SIZE = 0.1
    VAL_SIZE = 0.111
    RANDOM_STATE = 42

    # 打印/监控配置
    PRINT_MEMORY_USAGE = True
    PRINT_SAMPLE_DISTRIBUTION = True


config = Config()

# 创建目录
os.makedirs(config.DATASET_OUTPUT_DIR, exist_ok=True)
temp_chunk_dir = os.path.join(config.DATASET_OUTPUT_DIR, "temp_chunks")
os.makedirs(temp_chunk_dir, exist_ok=True)


# ===================== 辅助函数 =====================
def print_memory_usage(step_name=""):
    """打印当前内存占用"""
    if not Config.PRINT_MEMORY_USAGE:
        return
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"📊 内存占用 [{step_name}]：{mem_usage:.2f} GB")


def print_sample_distribution(modulation_samples, label_encoder_mapping):
    """打印各调制类型样本分布"""
    if not Config.PRINT_SAMPLE_DISTRIBUTION:
        return
    print("\n" + "-" * 70)
    print("📈 各调制类型样本分布汇总 (步长: {} | 50%重叠)".format(config.STRIDE))
    print(f"{'调制类型':<12} {'样本总数':<15} {'标签ID':<8} {'文件数':<8}")
    print("-" * 70)
    total_all = 0
    for mod, items in modulation_samples.items():
        mod_total = sum([item['file_info']['estimated_samples'] for item in items])
        total_all += mod_total
        label_id = label_encoder_mapping.get(mod, -1)
        file_count = len(items)
        print(f"{mod:<12} {mod_total:<15,} {label_id:<8} {file_count:<8}")
    print("-" * 70)
    print(f"{'总计':<12} {total_all:<15,} {'-':<8} {len([i for v in modulation_samples.values() for i in v]):<8}")
    print("-" * 70)


# ===================== 核心函数：文件解析 =====================
def get_file_iq_info(file_path):
    """
    解析单个文件的IQ基础信息，并根据步长估算样本数
    """
    try:
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        file_info = {
            'file_path': file_path,
            'filename': filename,
            'file_type': file_ext[1:],  # bin/wav
            'modulation': None,
            'sample_rate_hz': None,
            'total_iq_pairs': 0,
            'valid_iq_pairs': 0,
            'estimated_samples': 0,  # 根据步长计算的样本数
            'frame_size': None,
            'total_frames': None
        }

        # 提取调制类型
        name_without_ext = os.path.splitext(filename)[0]
        modulation_part = name_without_ext.split('_', 1)[0].strip()
        file_info['modulation'] = modulation_part

        # 提取采样率
        sample_rate_pattern = r'(\d+\.?\d*)\s*([kM]SPS)'
        sample_rate_match = re.search(sample_rate_pattern, name_without_ext)
        if sample_rate_match:
            num = float(sample_rate_match.group(1))
            unit = sample_rate_match.group(2)
            sample_rate = num * 1e3 if unit == 'kSPS' else num * 1e6
            file_info['sample_rate_hz'] = sample_rate

        # ========== BIN文件解析 ==========
        if file_ext == '.bin':
            frame_size_pattern = r'_(\d+)\.bin$'
            frame_size_match = re.search(frame_size_pattern, filename)
            if not frame_size_match:
                return None

            frame_size = int(frame_size_match.group(1))
            file_info['frame_size'] = frame_size

            # 仅仅读取元数据，不读取整个文件内容
            file_size_bytes = os.path.getsize(file_path)
            total_iq = file_size_bytes // 4  # int16 * 2 = 4 bytes

            file_info['total_iq_pairs'] = total_iq
            total_frames = total_iq // frame_size
            file_info['total_frames'] = total_frames
            file_info['valid_iq_pairs'] = total_frames * frame_size

            # 计算样本数 (适配 Stride)
            # 每帧的有效区间长度
            valid_len_per_frame = frame_size - config.SAMPLE_LENGTH + 1
            if valid_len_per_frame <= 0:
                return None

            # 每帧能切出的样本数 = (有效长度 / 步长) 向上取整
            samples_per_frame = (valid_len_per_frame + config.STRIDE - 1) // config.STRIDE
            file_info['estimated_samples'] = total_frames * samples_per_frame

        # ========== WAV文件解析 ==========
        elif file_ext == '.wav':
            file_size = os.path.getsize(file_path)
            # 跳过1068字节头，剩余字节 / 4 = IQ对数
            valid_bytes = file_size - 1068
            if valid_bytes <= 0: return None

            total_iq = valid_bytes // 4
            file_info['total_iq_pairs'] = total_iq
            file_info['valid_iq_pairs'] = total_iq

            # 计算样本数 (适配 Stride)
            if total_iq < config.SAMPLE_LENGTH:
                return None

            valid_len = total_iq - config.SAMPLE_LENGTH + 1
            # 样本数 = (有效长度 / 步长) 向上取整
            file_info['estimated_samples'] = (valid_len + config.STRIDE - 1) // config.STRIDE

        else:
            return None

        if file_info['estimated_samples'] <= 0:
            return None

        print(f"   └─ {filename} -> 预计生成样本: {file_info['estimated_samples']:,} (步长: {config.STRIDE})")
        return file_info

    except Exception as e:
        print(f"⚠️  处理文件失败：{file_path} -> {str(e)}")
        return None


def stream_read_sample(file_info, start_idx):
    """
    读取样本并转为 Float16
    """
    file_path = file_info['file_path'].replace('/', '\\')
    if not os.path.exists(file_path):
        file_path = os.path.join(config.DATA_DIR, os.path.basename(file_path))

    if file_info['file_type'] == 'bin':
        seek_pos = start_idx * 4
    elif file_info['file_type'] == 'wav':
        seek_pos = 1068 + start_idx * 4
    else:
        return np.zeros((config.SAMPLE_LENGTH, 2), dtype=config.DTYPE)

    try:
        with open(file_path, 'rb') as f:
            f.seek(seek_pos)
            data = np.fromfile(f, dtype=np.int16, count=config.SAMPLE_LENGTH * 2)
    except Exception as e:
        return np.zeros((config.SAMPLE_LENGTH, 2), dtype=config.DTYPE)

    if len(data) < config.SAMPLE_LENGTH * 2:
        sample = np.zeros((config.SAMPLE_LENGTH, 2), dtype=np.int16)
        valid_len = len(data) // 2
        sample[:valid_len] = data.reshape(-1, 2)
    else:
        sample = data.reshape(-1, 2)

    # 归一化并转为 float16
    sample_norm = sample.astype(config.DTYPE) / 32767.0
    return sample_norm


# ===================== 主函数 =====================
def construct_dataset_stream():
    total_start_time = time.time()

    print("=" * 80)
    print("🚀 开始流式构造调制信号数据集 (Float16 + 50% Overlap)")
    print(f"⚙️  配置：样本长={config.SAMPLE_LENGTH} | 步长={config.STRIDE} | 精度={config.DTYPE.__name__}")
    print("=" * 80)

    # 1. 统计文件
    print("\n📌 第一步：解析文件信息...")
    all_file_metadata = []
    modulation_samples = {}
    total_samples_estimated = 0

    for filename in os.listdir(config.DATA_DIR):
        file_path = os.path.join(config.DATA_DIR, filename)
        if not os.path.isfile(file_path): continue

        file_info = get_file_iq_info(file_path)
        if not file_info: continue

        all_file_metadata.append(file_info)
        mod = file_info['modulation']
        if mod not in modulation_samples: modulation_samples[mod] = []

        modulation_samples[mod].append({'file_info': file_info})
        total_samples_estimated += file_info['estimated_samples']

    # 保存元数据和标签映射
    all_modulations = sorted(list(modulation_samples.keys()))
    label_encoder_mapping = {mod: idx for idx, mod in enumerate(all_modulations)}

    # 打印分布
    print_sample_distribution(modulation_samples, label_encoder_mapping)

    # 保存 Label Mapping
    label_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_encoder_mapping, f, ensure_ascii=False, indent=4)

    # 2. 生成索引并分块
    print("\n📌 第二步：生成索引、划分并分块写入...")
    chunk_counter = {'train': 0, 'val': 0, 'test': 0}
    buffer = {
        'train': {'data': [], 'labels': []},
        'val': {'data': [], 'labels': []},
        'test': {'data': [], 'labels': []}
    }

    for modulation in tqdm(all_modulations, desc="处理调制类型"):
        mod_label = label_encoder_mapping[modulation]
        mod_files = modulation_samples[modulation]

        # 收集该调制类型下的所有样本索引
        mod_all_samples = []

        for file_item in mod_files:
            file_info = file_item['file_info']

            if file_info['file_type'] == 'bin':
                # BIN文件：按帧 + 步长 遍历
                frame_size = file_info['frame_size']
                total_frames = file_info['total_frames']

                # 每一帧的有效起始点范围
                max_start_in_frame = frame_size - config.SAMPLE_LENGTH + 1

                frame_start_addr = 0
                for _ in range(total_frames):
                    # 在帧内应用步长
                    for inner_start in range(0, max_start_in_frame, config.STRIDE):
                        global_start = frame_start_addr + inner_start
                        mod_all_samples.append((file_info, global_start))
                    frame_start_addr += frame_size

            else:
                # WAV文件：全程应用步长
                max_start = file_info['valid_iq_pairs'] - config.SAMPLE_LENGTH + 1
                for start_idx in range(0, max_start, config.STRIDE):
                    mod_all_samples.append((file_info, start_idx))

        # 划分数据集 (Train/Val/Test)
        if len(mod_all_samples) == 0: continue

        X_train_val, X_test = train_test_split(mod_all_samples, test_size=config.TEST_SIZE,
                                               random_state=config.RANDOM_STATE)
        X_train, X_val = train_test_split(X_train_val, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE)

        # 定义写入函数
        def process_split(sample_list, split_name):
            for item in sample_list:
                f_info, s_idx = item
                data = stream_read_sample(f_info, s_idx)

                buffer[split_name]['data'].append(data)
                buffer[split_name]['labels'].append(mod_label)

                # 缓冲区满 -> 写入
                if len(buffer[split_name]['data']) >= config.CHUNK_SIZE:
                    save_chunk(split_name)

        def save_chunk(split_name):
            data_arr = np.array(buffer[split_name]['data']).transpose(0, 2, 1)  # (N, 2, L)
            label_arr = np.array(buffer[split_name]['labels'])

            c_idx = chunk_counter[split_name]
            path = os.path.join(temp_chunk_dir, f"{split_name}_chunk_{c_idx}.npz")

            # 使用压缩保存以进一步减小体积
            np.savez_compressed(path, data=data_arr, labels=label_arr)

            buffer[split_name]['data'] = []
            buffer[split_name]['labels'] = []
            chunk_counter[split_name] += 1
            print_memory_usage(f"{split_name}_chunk_{c_idx}")

        # 执行写入
        process_split(X_train, 'train')
        process_split(X_val, 'val')
        process_split(X_test, 'test')

    # 3. 写入剩余数据
    print("\n📌 第三步：清理缓冲区...")
    for split in ['train', 'val', 'test']:
        if len(buffer[split]['data']) > 0:
            save_chunk(split)

    # 4. 合并分块 (修复了 Windows PermissionError 问题)
    print("\n📌 第四步：合并最终文件...")
    for split in ['train', 'val', 'test']:
        all_data = []
        all_labels = []
        count = chunk_counter[split]

        if count == 0: continue

        print(f"合并 {split} 集 ({count} 个分块)...")
        for i in tqdm(range(count)):
            p = os.path.join(temp_chunk_dir, f"{split}_chunk_{i}.npz")
            if os.path.exists(p):
                # ================= 关键修复 =================
                # 使用 with 上下文管理器，确保加载完自动关闭文件句柄
                try:
                    with np.load(p) as loaded:
                        all_data.append(loaded['data'])
                        all_labels.append(loaded['labels'])

                    # 此时文件已关闭，可以安全删除
                    os.remove(p)
                except Exception as e:
                    print(f"⚠️  警告：无法删除临时文件 {p} -> {e}")
                # ===========================================

        if all_data:
            final_data = np.concatenate(all_data, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)

            # 保存最终文件
            np.save(os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy"), final_data)
            np.save(os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy"), final_labels)

            data_size = final_data.nbytes / 1024 / 1024 / 1024
            print(f"✅ {split} 保存完成: 样本数 {len(final_data):,} | 大小 {data_size:.2f} GB")

    # 删除临时目录
    try:
        if os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)  # 强力删除整个文件夹
    except Exception as e:
        print(f"⚠️  清理临时目录失败: {e}")

    print("\n" + "=" * 80)
    print(f"🎉 处理完成！输出目录：{config.DATASET_OUTPUT_DIR}")
    print(f"🚀 最终步长：{config.STRIDE} (50% Overlap)")
    print(f"💾 数据类型：{config.DTYPE.__name__} (Float16)")
    print("=" * 80)


if __name__ == "__main__":
    construct_dataset_stream()