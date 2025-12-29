import os
import re
import json
import numpy as np
import pandas as pd


# -------------------------- 配置参数 --------------------------
class Config:
    DATA_DIR = "../../DataSet"  # 原始.bin/.wav文件目录
    METADATA_OUTPUT_DIR = "./modulation_metadata"  # 元数据输出目录
    SAMPLE_LENGTH = 4096  # 单个样本IQ对数量
    STEP = 1  # 滑动步长（固定为1）


# 初始化配置
config = Config()
os.makedirs(config.METADATA_OUTPUT_DIR, exist_ok=True)


# -------------------------- 核心工具函数 --------------------------
def get_file_iq_info(file_path):
    """获取文件的总IQ对数量和类型"""
    try:
        if file_path.endswith('.bin'):
            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.int16)
            total_iq = len(data) // 2  # IQ交替存储
            file_type = 'bin'
            # .bin文件按131072对IQ分帧
            frame_size = 131072
            total_frames = total_iq // frame_size
            valid_iq = total_frames * frame_size  # 仅保留完整帧
            num_samples_per_file = sum([frame_size - config.SAMPLE_LENGTH + 1 for _ in range(total_frames)])

        elif file_path.endswith('.wav'):
            with open(file_path, 'rb') as f:
                f.seek(1068)  # 跳过头部
                data = np.fromfile(f, dtype=np.int16)
            total_iq = len(data) // 2
            file_type = 'wav'
            valid_iq = total_iq
            num_samples_per_file = total_iq - config.SAMPLE_LENGTH + 1 if total_iq >= config.SAMPLE_LENGTH else 0

        else:
            return None

        # 解析调制类型和采样率
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
            'num_samples': num_samples_per_file,  # 该文件可生成的步长1样本数
            'sample_length': config.SAMPLE_LENGTH,
            'step': config.STEP
        }
    except Exception as e:
        print(f"⚠️  处理文件失败：{file_path} -> {str(e)}")
        return None


# -------------------------- 生成全局元数据 --------------------------
def generate_metadata():
    print("=" * 70)
    print("🚀 生成原始文件元数据（用于动态滑动窗口加载）")
    print(f"📌 配置：4096对IQ/样本 | 滑动步长=1")
    print("=" * 70)

    # 遍历所有文件，生成元数据
    all_file_metadata = []
    global_sample_counter = 0  # 全局样本索引（唯一标识每个滑动窗口样本）
    global_sample_mapping = []  # 全局样本索引 → 文件+起始位置映射

    for filename in os.listdir(config.DATA_DIR):
        file_path = os.path.join(config.DATA_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        file_info = get_file_iq_info(file_path)
        if not file_info or file_info['num_samples'] == 0:
            continue

        all_file_metadata.append(file_info)

        # 生成该文件的所有样本映射（全局索引→文件内起始位置）
        if file_info['file_type'] == 'bin':
            # .bin文件：逐帧生成映射
            frame_size = 131072
            frame_start = 0
            for frame_idx in range(file_info['valid_iq_pairs'] // frame_size):
                frame_samples = frame_size - config.SAMPLE_LENGTH + 1
                for frame_inner_start in range(frame_samples):
                    # 全局样本索引 → (文件路径, 全局起始IQ位置, 调制类型)
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
            # .wav文件：连续生成映射
            for start_iq_idx in range(file_info['num_samples']):
                global_sample_mapping.append({
                    'global_idx': global_sample_counter,
                    'file_path': file_path,
                    'start_iq_idx': start_iq_idx,
                    'modulation': file_info['modulation']
                })
                global_sample_counter += 1

        print(f"📄 处理完成：{filename} → 可生成{file_info['num_samples']}个步长1样本")

    # 保存元数据
    # 1. 文件级元数据
    pd.DataFrame(all_file_metadata).to_csv(
        os.path.join(config.METADATA_OUTPUT_DIR, "file_metadata.csv"),
        index=False, encoding='utf-8'
    )

    # 2. 全局样本映射（核心：用于动态加载）
    pd.DataFrame(global_sample_mapping).to_csv(
        os.path.join(config.METADATA_OUTPUT_DIR, "global_sample_mapping.csv"),
        index=False, encoding='utf-8'
    )

    # 3. 调制类型编码
    all_modulations = sorted(list(set([f['modulation'] for f in all_file_metadata])))
    label_encoder_mapping = {mod: idx for idx, mod in enumerate(all_modulations)}
    with open(os.path.join(config.METADATA_OUTPUT_DIR, "label_mapping.json"), 'w') as f:
        json.dump({
            'label_to_idx': label_encoder_mapping,
            'idx_to_label': {v: k for k, v in label_encoder_mapping.items()},
            'total_samples': global_sample_counter
        }, f, indent=4)

    # 输出统计信息
    print("\n" + "=" * 70)
    print("🎉 元数据生成完成！")
    print(f"📊 统计信息：")
    print(f"  - 有效文件数：{len(all_file_metadata)}")
    print(f"  - 总步长1样本数：{global_sample_counter}")
    print(f"  - 调制类型数：{len(all_modulations)}")
    print(f"  - 元数据保存目录：{config.METADATA_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_metadata()