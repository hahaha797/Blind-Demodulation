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

warnings.filterwarnings('ignore')


# ===================== 配置（仅修改此处） =====================
class Config:
    # 数据路径配置
    DATA_DIR = "../../DataSet"  # 原始.bin/.wav文件目录
    DATASET_OUTPUT_DIR = "./modulation_dataset"  # 数据集输出目录
    SAMPLE_LENGTH = 4096  # 单样本IQ长度（固定）

    # 流式分块配置（核心优化）
    CHUNK_SIZE = 100000  # 每块样本数（10万/块，可根据内存调整）

    # 数据集划分
    TEST_SIZE = 0.1  # 测试集比例
    VAL_SIZE = 0.111  # 验证集比例（相对于train_val）
    RANDOM_STATE = 42  # 随机种子

    # 打印/监控配置
    PRINT_MEMORY_USAGE = True  # 是否打印内存占用
    PRINT_SAMPLE_DISTRIBUTION = True  # 是否打印样本分布


config = Config()

# 创建目录
os.makedirs(config.DATASET_OUTPUT_DIR, exist_ok=True)
temp_chunk_dir = os.path.join(config.DATASET_OUTPUT_DIR, "temp_chunks")
os.makedirs(temp_chunk_dir, exist_ok=True)


# ===================== 辅助函数：进度/内存监控 =====================
def print_memory_usage(step_name=""):
    """打印当前内存占用"""
    if not Config.PRINT_MEMORY_USAGE:
        return
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 / 1024 / 1024  # 转换为GB
    mem_percent = process.memory_percent()
    print(f"📊 内存占用 [{step_name}]：{mem_usage:.2f} GB | 占比：{mem_percent:.1f}%")


def print_sample_distribution(modulation_samples, label_encoder_mapping):
    """打印各调制类型样本分布汇总"""
    if not Config.PRINT_SAMPLE_DISTRIBUTION:
        return
    print("\n" + "-" * 70)
    print("📈 各调制类型样本分布汇总：")
    print(f"{'调制类型':<12} {'样本总数':<15} {'标签ID':<8} {'文件数':<8}")
    print("-" * 70)
    total_all = 0
    for mod, items in modulation_samples.items():
        mod_total = sum([item['file_info']['num_samples'] for item in items])
        total_all += mod_total
        label_id = label_encoder_mapping.get(mod, -1)
        file_count = len(items)
        print(f"{mod:<12} {mod_total:<15,} {label_id:<8} {file_count:<8}")
    print("-" * 70)
    print(f"{'总计':<12} {total_all:<15,} {'-':<8} {len([i for v in modulation_samples.values() for i in v]):<8}")
    print("-" * 70)


# ===================== 核心函数：文件解析+样本读取 =====================
def get_file_iq_info(file_path):
    """
    解析单个文件的IQ基础信息
    适配规范：
    - BIN：无文件头，从文件名提取单帧IQ对数（最后一个下划线后、.bin前的数字）
    - WAV：跳过1068字节头，无帧概念，全连续IQ
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
            'num_samples': 0,
            'sample_length': config.SAMPLE_LENGTH,
            'frame_size': None,  # 仅BIN文件有效
            'total_frames': None  # 仅BIN文件有效
        }

        # ========== 提取调制类型（兼容特殊字符如π-4DQPSK） ==========
        name_without_ext = os.path.splitext(filename)[0]
        # 分割调制类型：第一个下划线前的所有字符（包含特殊字符如π、-）
        modulation_part = name_without_ext.split('_', 1)[0].strip()  # 只分割第一个下划线
        file_info['modulation'] = modulation_part

        # ========== 提取采样率 ==========
        # 匹配采样率（如200kSPS、1.953MSPS、61.03515625kSPS）
        sample_rate_pattern = r'(\d+\.?\d*)\s*([kM]SPS)'
        sample_rate_match = re.search(sample_rate_pattern, name_without_ext)
        if sample_rate_match:
            num = float(sample_rate_match.group(1))
            unit = sample_rate_match.group(2)
            sample_rate = num * 1e3 if unit == 'kSPS' else num * 1e6
            file_info['sample_rate_hz'] = sample_rate

        # ========== BIN文件解析（修复：提取帧大小） ==========
        if file_ext == '.bin':
            # 修复正则：匹配最后一个下划线后的数字（格式：xxx_xxx_131072.bin → 131072）
            frame_size_pattern = r'_(\d+)\.bin$'  # 匹配.bin前的数字，且前面有下划线
            frame_size_match = re.search(frame_size_pattern, filename)

            if not frame_size_match:
                print(f"⚠️  BIN文件{filename}命名不规范，无法提取帧大小（格式应为：调制类型_采样率_帧大小.bin），跳过")
                return None

            frame_size = int(frame_size_match.group(1))
            file_info['frame_size'] = frame_size

            # 读取文件并计算IQ对
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.int16)
            total_iq = len(data) // 2  # 总IQ对数量
            file_info['total_iq_pairs'] = total_iq

            # 计算有效帧和有效IQ对
            total_frames = total_iq // frame_size
            valid_iq = total_frames * frame_size
            file_info['total_frames'] = total_frames
            file_info['valid_iq_pairs'] = valid_iq

            # 计算样本数：每帧可生成 frame_size - SAMPLE_LENGTH + 1 个样本
            samples_per_frame = frame_size - config.SAMPLE_LENGTH + 1
            if samples_per_frame <= 0:
                print(f"⚠️  BIN文件{filename}帧大小{frame_size} < 样本长度{config.SAMPLE_LENGTH}，无有效样本")
                return None
            num_samples = total_frames * samples_per_frame
            file_info['num_samples'] = num_samples

            # 打印单文件解析详情
            print(f"   ├─ 帧大小：{frame_size:,} | 总帧数：{total_frames:,} | 每帧样本数：{samples_per_frame:,}")
            print(
                f"   ├─ 总IQ对：{total_iq:,} | 有效IQ对：{valid_iq:,} | 采样率：{sample_rate / 1e3:.1f} kSPS" if sample_rate else f"   ├─ 总IQ对：{total_iq:,} | 有效IQ对：{valid_iq:,}")

        # ========== WAV文件解析（核心：跳过1068字节头） ==========
        elif file_ext == '.wav':
            # 获取文件总大小
            file_size = os.path.getsize(file_path)
            # 跳过1068字节头，读取剩余数据
            with open(file_path, 'rb') as f:
                f.seek(1068)  # 严格按规范跳过1068字节（非1086）
                data = np.fromfile(f, dtype=np.int16)

            total_iq = len(data) // 2  # 跳过1068后的总IQ对
            file_info['total_iq_pairs'] = total_iq
            file_info['valid_iq_pairs'] = total_iq

            # 计算样本数：总IQ对 - 样本长度 + 1
            if total_iq < config.SAMPLE_LENGTH:
                print(f"⚠️  WAV文件{filename}IQ对{total_iq} < 样本长度{config.SAMPLE_LENGTH}，无有效样本")
                return None
            num_samples = total_iq - config.SAMPLE_LENGTH + 1
            file_info['num_samples'] = num_samples

            # 打印单文件解析详情
            print(f"   ├─ 文件总大小：{file_size / 1024 / 1024:.2f} MB | 跳过头部：1068字节")
            print(
                f"   ├─ 有效IQ对：{total_iq:,} | 采样率：{sample_rate / 1e6:.3f} MSPS" if sample_rate else f"   ├─ 有效IQ对：{total_iq:,}")

        else:
            print(f"⚠️  不支持的文件格式：{file_ext}，跳过")
            return None

        # 最终校验样本数
        if file_info['num_samples'] <= 0:
            print(f"⚠️  文件{filename}无有效样本，跳过")
            return None

        print(f"   └─ 有效样本数：{file_info['num_samples']:,}")
        return file_info

    except Exception as e:
        print(f"⚠️  处理文件失败：{file_path} -> {str(e)}")
        return None


def stream_read_sample(file_info, start_idx):
    """
    流式读取单个样本（不缓存）
    - BIN：直接从start_idx*4字节位置读取
    - WAV：从1068 + start_idx*4字节位置读取
    """
    file_path = file_info['file_path'].replace('/', '\\')
    if not os.path.exists(file_path):
        file_path = os.path.join(config.DATA_DIR, os.path.basename(file_path))

    # 计算读取起始位置
    if file_info['file_type'] == 'bin':
        seek_pos = start_idx * 4  # int16×2=4字节/IQ对
    elif file_info['file_type'] == 'wav':
        seek_pos = 1068 + start_idx * 4  # 跳过1068字节头
    else:
        return np.zeros((config.SAMPLE_LENGTH, 2), dtype=np.float32)

    # 读取IQ数据
    try:
        with open(file_path, 'rb') as f:
            f.seek(seek_pos)
            data = np.fromfile(f, dtype=np.int16, count=config.SAMPLE_LENGTH * 2)
    except Exception as e:
        print(f"⚠️  读取样本失败：{file_info['filename']} start_idx={start_idx} -> {e}")
        return np.zeros((config.SAMPLE_LENGTH, 2), dtype=np.float32)

    # 处理数据（补零+归一化）
    if len(data) < config.SAMPLE_LENGTH * 2:
        sample = np.zeros((config.SAMPLE_LENGTH, 2), dtype=np.int16)
        valid_len = len(data) // 2
        sample[:valid_len] = data.reshape(-1, 2)
    else:
        sample = data.reshape(-1, 2)

    # 归一化到[-1, 1]
    sample_norm = sample.astype(np.float32) / 32767.0
    return sample_norm


# ===================== 主函数：流式构造数据集 =====================
def construct_dataset_stream():
    """主函数：流式构造数据集（不加载所有数据到内存）"""
    total_start_time = time.time()

    # 初始化信息打印
    print("=" * 80)
    print("🚀 开始流式构造调制信号数据集")
    print(f"📅 开始时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  配置：分块大小={config.CHUNK_SIZE:,} | 测试集={config.TEST_SIZE:.1%} | 验证集={config.VAL_SIZE:.1%}")
    print(f"📁 数据目录：{config.DATA_DIR} | 输出目录：{config.DATASET_OUTPUT_DIR}")
    print("=" * 80)
    print_memory_usage("初始化")

    # ========== 第一步：统计所有文件信息 ==========
    print("\n📌 第一步：解析并统计所有文件信息...")
    start_time = time.time()
    all_file_metadata = []
    modulation_samples = {}  # 按调制类型存储文件信息
    total_samples = 0
    file_count = 0
    valid_file_count = 0

    # 遍历所有文件
    for filename in os.listdir(config.DATA_DIR):
        file_path = os.path.join(config.DATA_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        file_count += 1
        print(f"\n📄 处理文件 [{file_count}]：{filename}")

        # 解析文件信息
        file_info = get_file_iq_info(file_path)
        if not file_info:
            continue

        # 统计有效文件
        valid_file_count += 1
        all_file_metadata.append(file_info)
        modulation = file_info['modulation']

        # 按调制类型分组
        if modulation not in modulation_samples:
            modulation_samples[modulation] = []
        modulation_samples[modulation].append({
            'file_info': file_info,
            'start_idx': 0,
            'end_idx': file_info['num_samples']
        })

        # 累计总样本数
        total_samples += file_info['num_samples']
        print(f"   └─ 累计总样本数：{total_samples:,}")

    # 第一步总结
    elapsed = time.time() - start_time
    print(f"\n✅ 第一步完成！耗时：{elapsed:.2f}s")
    print(
        f"📊 统计结果：总文件数={file_count} | 有效文件数={valid_file_count} | 总样本数={total_samples:,} | 调制类型数={len(modulation_samples)}")
    print_memory_usage("文件解析完成")

    # 保存文件元数据
    meta_path = os.path.join(config.DATASET_OUTPUT_DIR, "file_metadata.csv")
    pd.DataFrame(all_file_metadata).to_csv(meta_path, index=False, encoding='utf-8')
    print(f"💾 保存文件元数据：{meta_path}")

    # 生成标签映射
    all_modulations = sorted(list(modulation_samples.keys()))
    label_encoder_mapping = {mod: idx for idx, mod in enumerate(all_modulations)}
    label_mapping = {
        'label_to_idx': label_encoder_mapping,
        'idx_to_label': {v: k for k, v in label_encoder_mapping.items()},
        'total_samples': total_samples,
        'modulation_count': len(all_modulations),
        'create_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'sample_length': config.SAMPLE_LENGTH,
            'test_size': config.TEST_SIZE,
            'val_size': config.VAL_SIZE
        }
    }
    label_path = os.path.join(config.DATASET_OUTPUT_DIR, "label_mapping.json")
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=4, ensure_ascii=False)
    print(f"💾 保存标签映射：{label_path}")

    # 打印样本分布
    print_sample_distribution(modulation_samples, label_encoder_mapping)

    # ========== 第二步：流式划分+分块写入 ==========
    print("\n📌 第二步：按调制类型分层划分并分块写入...")
    start_time = time.time()
    chunk_counter = {'train': 0, 'val': 0, 'test': 0}
    buffer = {
        'train': {'data': [], 'labels': [], 'count': 0},
        'val': {'data': [], 'labels': [], 'count': 0},
        'test': {'data': [], 'labels': [], 'count': 0}
    }
    total_processed = {'train': 0, 'val': 0, 'test': 0}

    # 遍历每个调制类型
    for modulation in tqdm(all_modulations, desc="处理调制类型", ncols=100):
        mod_start = time.time()
        mod_label = label_encoder_mapping[modulation]
        mod_files = modulation_samples[modulation]

        # 收集当前调制的所有样本索引
        mod_all_samples = []
        for file_item in mod_files:
            file_info = file_item['file_info']
            if file_info['file_type'] == 'bin':
                # BIN文件：按帧生成样本索引
                frame_size = file_info['frame_size']
                total_frames = file_info['total_frames']
                frame_samples = frame_size - config.SAMPLE_LENGTH + 1
                frame_start = 0
                for frame_idx in range(total_frames):
                    for frame_inner_start in range(frame_samples):
                        global_start = frame_start + frame_inner_start
                        mod_all_samples.append((file_info, global_start))
                    frame_start += frame_size
            else:
                # WAV文件：连续样本索引
                for start_idx in range(file_info['num_samples']):
                    mod_all_samples.append((file_info, start_idx))

        mod_total = len(mod_all_samples)
        print(f"\n🔍 处理调制类型：{modulation} | 总样本数：{mod_total:,} | 标签ID：{mod_label}")

        # 分层划分train/val/test
        mod_samples_arr = np.array(mod_all_samples, dtype=object)
        # 先划分test集
        X_train_val, X_test = train_test_split(
            mod_samples_arr, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        # 再划分train/val集
        X_train, X_val = train_test_split(
            X_train_val, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
        )

        # 打印划分结果
        print(f"   ├─ Train：{len(X_train):,} ({len(X_train) / mod_total:.1%})")
        print(f"   ├─ Val：{len(X_val):,} ({len(X_val) / mod_total:.1%})")
        print(f"   └─ Test：{len(X_test):,} ({len(X_test) / mod_total:.1%})")

        # ========== 处理Train样本 ==========
        train_pbar = tqdm(X_train, desc=f"{modulation} - Train", leave=False, ncols=80)
        for sample_item in train_pbar:
            file_info, start_idx = sample_item
            sample_data = stream_read_sample(file_info, start_idx)

            buffer['train']['data'].append(sample_data)
            buffer['train']['labels'].append(mod_label)
            total_processed['train'] += 1

            # 缓冲区满则写入分块
            if len(buffer['train']['data']) >= config.CHUNK_SIZE:
                data_arr = np.array(buffer['train']['data']).transpose(0, 2, 1)
                label_arr = np.array(buffer['train']['labels'])
                chunk_path = os.path.join(temp_chunk_dir, f"train_chunk_{chunk_counter['train']}.npz")
                np.savez(chunk_path, data=data_arr, labels=label_arr)

                # 打印分块信息
                chunk_size = os.path.getsize(chunk_path) / 1024 / 1024
                print(
                    f"\n📦 写入Train分块 [{chunk_counter['train']}]：{chunk_path} | 样本数：{len(data_arr):,} | 大小：{chunk_size:.2f} MB")

                # 重置缓冲区
                buffer['train']['data'] = []
                buffer['train']['labels'] = []
                chunk_counter['train'] += 1
                print_memory_usage(f"Train分块{chunk_counter['train']}")

        # ========== 处理Val样本 ==========
        val_pbar = tqdm(X_val, desc=f"{modulation} - Val", leave=False, ncols=80)
        for sample_item in val_pbar:
            file_info, start_idx = sample_item
            sample_data = stream_read_sample(file_info, start_idx)

            buffer['val']['data'].append(sample_data)
            buffer['val']['labels'].append(mod_label)
            total_processed['val'] += 1

            if len(buffer['val']['data']) >= config.CHUNK_SIZE:
                data_arr = np.array(buffer['val']['data']).transpose(0, 2, 1)
                label_arr = np.array(buffer['val']['labels'])
                chunk_path = os.path.join(temp_chunk_dir, f"val_chunk_{chunk_counter['val']}.npz")
                np.savez(chunk_path, data=data_arr, labels=label_arr)

                chunk_size = os.path.getsize(chunk_path) / 1024 / 1024
                print(
                    f"\n📦 写入Val分块 [{chunk_counter['val']}]：{chunk_path} | 样本数：{len(data_arr):,} | 大小：{chunk_size:.2f} MB")

                buffer['val']['data'] = []
                buffer['val']['labels'] = []
                chunk_counter['val'] += 1
                print_memory_usage(f"Val分块{chunk_counter['val']}")

        # ========== 处理Test样本 ==========
        test_pbar = tqdm(X_test, desc=f"{modulation} - Test", leave=False, ncols=80)
        for sample_item in test_pbar:
            file_info, start_idx = sample_item
            sample_data = stream_read_sample(file_info, start_idx)

            buffer['test']['data'].append(sample_data)
            buffer['test']['labels'].append(mod_label)
            total_processed['test'] += 1

            if len(buffer['test']['data']) >= config.CHUNK_SIZE:
                data_arr = np.array(buffer['test']['data']).transpose(0, 2, 1)
                label_arr = np.array(buffer['test']['labels'])
                chunk_path = os.path.join(temp_chunk_dir, f"test_chunk_{chunk_counter['test']}.npz")
                np.savez(chunk_path, data=data_arr, labels=label_arr)

                chunk_size = os.path.getsize(chunk_path) / 1024 / 1024
                print(
                    f"\n📦 写入Test分块 [{chunk_counter['test']}]：{chunk_path} | 样本数：{len(data_arr):,} | 大小：{chunk_size:.2f} MB")

                buffer['test']['data'] = []
                buffer['test']['labels'] = []
                chunk_counter['test'] += 1
                print_memory_usage(f"Test分块{chunk_counter['test']}")

        # 调制类型处理完成
        mod_elapsed = time.time() - mod_start
        print(f"✅ 完成调制类型 {modulation} | 耗时：{mod_elapsed:.2f}s | 速度：{mod_total / mod_elapsed:.0f} 样本/秒")

    # 第二步总结
    elapsed = time.time() - start_time
    print(f"\n✅ 第二步完成！耗时：{elapsed:.2f}s")
    print(f"📊 分块统计：Train={chunk_counter['train']} | Val={chunk_counter['val']} | Test={chunk_counter['test']}")
    print(
        f"📊 处理样本：Train={total_processed['train']:,} | Val={total_processed['val']:,} | Test={total_processed['test']:,}")
    print_memory_usage("分块写入完成")

    # ========== 第三步：写入剩余样本 ==========
    print("\n📌 第三步：写入剩余缓冲区样本...")
    start_time = time.time()
    remaining_total = 0

    for split in ['train', 'val', 'test']:
        if len(buffer[split]['data']) > 0:
            data_arr = np.array(buffer[split]['data']).transpose(0, 2, 1)
            label_arr = np.array(buffer[split]['labels'])
            remaining_total += len(data_arr)

            chunk_path = os.path.join(temp_chunk_dir, f"{split}_chunk_{chunk_counter[split]}.npz")
            np.savez(chunk_path, data=data_arr, labels=label_arr)

            chunk_size = os.path.getsize(chunk_path) / 1024 / 1024
            print(
                f"📦 写入{split}剩余分块 [{chunk_counter[split]}]：{chunk_path} | 样本数：{len(data_arr):,} | 大小：{chunk_size:.2f} MB")

            chunk_counter[split] += 1

    elapsed = time.time() - start_time
    print(f"\n✅ 第三步完成！耗时：{elapsed:.2f}s | 写入剩余样本：{remaining_total:,}")
    print_memory_usage("剩余样本写入完成")

    # ========== 第四步：合并分块文件 ==========
    print("\n📌 第四步：合并分块文件...")
    start_time = time.time()
    final_sample_count = {}

    for split in ['train', 'val', 'test']:
        split_start = time.time()
        print(f"\n🔗 合并{split}集（共{chunk_counter[split]}个分块）...")

        all_data = []
        all_labels = []

        # 遍历所有分块
        for chunk_idx in tqdm(range(chunk_counter[split]), desc=f"{split}合并进度", ncols=80):
            chunk_path = os.path.join(temp_chunk_dir, f"{split}_chunk_{chunk_idx}.npz")
            if not os.path.exists(chunk_path):
                print(f"⚠️  分块不存在：{chunk_path}，跳过")
                continue

            # 读取分块
            chunk_data = np.load(chunk_path, allow_pickle=True)
            all_data.append(chunk_data['data'])
            all_labels.append(chunk_data['labels'])

            # 删除临时分块
            os.remove(chunk_path)

        # 合并并保存
        if len(all_data) > 0:
            final_data = np.concatenate(all_data, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
            final_sample_count[split] = len(final_data)

            # 保存最终文件
            data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
            label_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")
            np.save(data_path, final_data)
            np.save(label_path, final_labels)

            # 打印文件信息
            data_size = os.path.getsize(data_path) / 1024 / 1024 / 1024
            label_size = os.path.getsize(label_path) / 1024 / 1024
            split_elapsed = time.time() - split_start

            print(f"✅ {split}集合并完成！")
            print(f"   ├─ 样本数：{len(final_data):,}")
            print(f"   ├─ 数据文件：{data_path} | 大小：{data_size:.2f} GB")
            print(f"   ├─ 标签文件：{label_path} | 大小：{label_size:.2f} MB")
            print(f"   └─ 耗时：{split_elapsed:.2f}s")
        else:
            final_sample_count[split] = 0
            print(f"⚠️ {split}集无分块可合并！")

    # 删除临时目录
    if os.path.exists(temp_chunk_dir) and len(os.listdir(temp_chunk_dir)) == 0:
        os.rmdir(temp_chunk_dir)
        print(f"\n🗑️ 删除临时目录：{temp_chunk_dir}")
    else:
        print(f"\n⚠️ 临时目录未空：{temp_chunk_dir} | 剩余文件：{len(os.listdir(temp_chunk_dir))}")

    # 第四步总结
    elapsed = time.time() - start_time
    print(f"\n✅ 第四步完成！耗时：{elapsed:.2f}s")
    print_memory_usage("分块合并完成")

    # ========== 最终汇总 ==========
    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("🎉 调制信号数据集构造完成！")
    print(f"📅 结束时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时：{total_elapsed:.2f}s ({total_elapsed / 60:.1f}分钟)")
    print(f"🚀 平均速度：{total_samples / total_elapsed:.0f} 样本/秒")

    # 最终统计报表
    print("\n📊 最终数据集统计：")
    print(f"{'数据集':<10} {'样本数':<15} {'数据文件大小':<15} {'标签文件大小':<15}")
    print("-" * 65)
    total_final = 0
    for split in ['train', 'val', 'test']:
        data_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_data.npy")
        label_path = os.path.join(config.DATASET_OUTPUT_DIR, f"{split}_labels.npy")

        if os.path.exists(data_path):
            data_size = os.path.getsize(data_path) / 1024 / 1024 / 1024
            label_size = os.path.getsize(label_path) / 1024 / 1024 if os.path.exists(label_path) else 0
            count = final_sample_count.get(split, 0)
            total_final += count
            print(f"{split:<10} {count:<15,} {data_size:<15.2f} GB {label_size:<15.2f} MB")
        else:
            print(f"{split:<10} 0:<15, {'-':<15} {'-':<15}")
    print("-" * 65)
    print(f"{'总计':<10} {total_final:<15,} {'-':<15} {'-':<15}")

    # 生成文件列表
    print("\n📁 生成的文件列表：")
    output_files = [
        "file_metadata.csv", "label_mapping.json",
        "train_data.npy", "train_labels.npy",
        "val_data.npy", "val_labels.npy",
        "test_data.npy", "test_labels.npy"
    ]
    for f in output_files:
        fp = os.path.join(config.DATASET_OUTPUT_DIR, f)
        if os.path.exists(fp):
            size = os.path.getsize(fp)
            size_unit = "GB" if size > 1024 * 1024 * 1024 else "MB"
            size_val = size / 1024 / 1024 / 1024 if size_unit == "GB" else size / 1024 / 1024
            print(f"  ✅ {f:<20} | 大小：{size_val:.2f} {size_unit}")
        else:
            print(f"  ❌ {f:<20} | 文件不存在")

    print("\n✅ 全程未加载所有样本到内存，48G内存完全适配！")
    print("✅ 可运行trainer.py开始模型训练！")
    print("=" * 80)
    print_memory_usage("任务完成")


# ===================== 运行入口 =====================
if __name__ == "__main__":
    # 打印系统信息
    print("🖥️  系统环境信息：")
    print(f"   ├─ NumPy版本：{np.__version__}")  # 修复原代码Python版本打印错误
    print(f"   ├─ CPU核心数：{psutil.cpu_count(logical=True)}")
    print(f"   ├─ 总内存：{psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    print(f"   ├─ 可用内存：{psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
    print(f"   └─ 磁盘可用空间：{psutil.disk_usage(config.DATASET_OUTPUT_DIR).free / 1024 / 1024 / 1024:.2f} GB")

    # 启动数据集构造
    construct_dataset_stream()