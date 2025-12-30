import os
import json
import time
import re
import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


# ===================== 配置 =====================
class Config:
    # 模型和映射文件路径 (对应训练时的输出目录)
    MODEL_PATH = "../src/modulation_dataset_50overlap/best_model.pth"
    MAPPING_PATH = "../src/modulation_dataset_50overlap/label_mapping.json"

    # 原始数据目录 (你要验证的文件夹)
    RAW_DATA_DIR = "../../DataSet"

    # 采样参数
    SAMPLE_LENGTH = 4096
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()


# ===================== 1. 模型定义 (必须与训练代码一致) =====================
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


def get_activation():
    return nn.SiLU() if hasattr(nn, 'SiLU') else Swish()


class YOLO12_1D_Modulation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), get_activation(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), get_activation(),
            nn.Dropout1d(0.1),
            nn.Conv1d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128), get_activation(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256), get_activation(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), get_activation(),
            nn.Conv1d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512), get_activation(),
        )
        # Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            get_activation(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ===================== 2. 原始文件预处理 =====================
def load_and_preprocess_raw_file(file_path, sample_length=4096):
    """
    读取原始 .bin 或 .wav 文件，提取中间一段进行归一化处理
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    try:
        # 1. 读取二进制数据
        if ext == '.bin':
            # BIN文件：直接读取
            with open(file_path, 'rb') as f:
                raw_data = np.fromfile(f, dtype=np.int16)
        elif ext == '.wav':
            # WAV文件：跳过1068字节头
            with open(file_path, 'rb') as f:
                f.seek(1068)
                raw_data = np.fromfile(f, dtype=np.int16)
        else:
            return None, "不支持的文件格式"

        # 2. 检查长度
        iq_pairs = len(raw_data) // 2
        if iq_pairs < sample_length:
            return None, f"样本过短 ({iq_pairs} < {sample_length})"

        # 3. 截取中间一段 (避免文件头尾的噪声)
        # 或者你可以改为随机截取 start = np.random.randint(0, iq_pairs - sample_length)
        start_idx = (iq_pairs - sample_length) // 2
        seek_pos = start_idx * 2  # int16数组索引

        extracted = raw_data[seek_pos: seek_pos + sample_length * 2]

        # 4. Reshape [L, 2] -> Transpose [2, L]
        iq_data = extracted.reshape(-1, 2).T

        # 5. 归一化 (int16 -> float32 [-1, 1])
        iq_data = iq_data.astype(np.float32) / 32767.0

        # 6. 转 Tensor 并增加 Batch 维度 [1, 2, 4096]
        input_tensor = torch.from_numpy(iq_data).unsqueeze(0)

        return input_tensor, "Success"

    except Exception as e:
        return None, str(e)


# ===================== 3. 推理核心逻辑 =====================
class InferenceEngine:
    def __init__(self):
        print(f"⏳ 正在初始化模型 (Device: {config.DEVICE})...")

        # 加载标签映射
        with open(config.MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # 兼容旧版json格式
            if 'label_to_idx' in mapping:
                self.idx_to_label = {v: k for k, v in mapping['label_to_idx'].items()}
            else:
                self.idx_to_label = {v: k for k, v in mapping.items()}
            self.num_classes = len(self.idx_to_label)

        # 初始化模型
        self.model = YOLO12_1D_Modulation(num_classes=self.num_classes).to(config.DEVICE)

        # 加载权重
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        # 如果保存的是整个checkpoint字典
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            acc = checkpoint.get('best_val_acc', 0.0)
            print(f"✅ 模型加载成功 (Val Acc: {acc:.2%})")
        else:
            # 如果只保存了state_dict
            self.model.load_state_dict(checkpoint)
            print("✅ 模型权重加载成功")

        self.model.eval()

    def predict(self, file_path):
        input_tensor, msg = load_and_preprocess_raw_file(file_path, config.SAMPLE_LENGTH)
        if input_tensor is None:
            return None, msg

        input_tensor = input_tensor.to(config.DEVICE)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)  # 计算概率

        elapsed = (time.time() - start_time) * 1000  # ms

        # 获取 Top-3 结果
        topk_probs, topk_indices = torch.topk(probs, 3)

        results = []
        for i in range(3):
            idx = topk_indices[0][i].item()
            prob = topk_probs[0][i].item()
            label = self.idx_to_label.get(idx, "Unknown")
            results.append((label, prob))

        return results, elapsed


# ===================== 主程序 =====================
if __name__ == "__main__":
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ 找不到模型文件: {config.MODEL_PATH}")
        exit()

    engine = InferenceEngine()

    # 获取所有原始文件
    files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.endswith(('.bin', '.wav'))]
    if not files:
        print(f"❌ 目录 {config.RAW_DATA_DIR} 下没有找到 .bin 或 .wav 文件")
        exit()

    # 随机选取 10 个文件进行测试
    import random

    random.shuffle(files)
    test_files = files[:10]

    print("\n" + "=" * 80)
    print(f"🚀 开始验证 (随机抽取 {len(test_files)} 个原始文件)")
    print("=" * 80)
    print(f"{'文件名':<35} | {'真实标签 (Guess)':<15} | {'预测结果 (Top-1)':<15} | {'置信度':<8} | {'结果':<5}")
    print("-" * 95)

    correct_count = 0

    for filename in test_files:
        file_path = os.path.join(config.RAW_DATA_DIR, filename)

        # 从文件名猜测真实标签 (假设文件名格式为 "QPSK_xxxx.bin")
        ground_truth = filename.split('_')[0]

        results, elapsed = engine.predict(file_path)

        if results:
            top1_label, top1_prob = results[0]

            # 简单判断对错 (不区分大小写)
            is_correct = ground_truth.lower() in top1_label.lower() or top1_label.lower() in ground_truth.lower()
            status = "✅" if is_correct else "❌"
            if is_correct: correct_count += 1

            # 格式化输出
            fname_short = (filename[:32] + '..') if len(filename) > 32 else filename
            print(f"{fname_short:<35} | {ground_truth:<15} | {top1_label:<15} | {top1_prob:.1%} | {status}")

            # 如果错误，显示 Top-2 和 Top-3
            if not is_correct:
                print(
                    f"   ↳ Top-2: {results[1][0]} ({results[1][1]:.1%}) | Top-3: {results[2][0]} ({results[2][1]:.1%})")
        else:
            print(f"{filename:<35} | 读取失败: {elapsed}")

    print("-" * 95)
    print(f"📊 统计: 正确 {correct_count}/{len(test_files)} | 准确率: {correct_count / len(test_files):.1%}")