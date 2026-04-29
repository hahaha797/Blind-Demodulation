import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fftpack import fftshift


# ===================== 0. 字体配置函数 (新增) =====================
def setup_font():
    # 核心字体配置：英文/数字用Times New Roman，中文用宋体，跨系统兼容
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [
        'Times New Roman',
        'SimSun',
        'Songti SC',
        'DejaVu Serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # 完全保留你的字号设置
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['xtick.labelsize'] = 26
    plt.rcParams['ytick.labelsize'] = 26
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 22

    print("✅ Font initialized: English=Times New Roman, Chinese=SimSun")


# ===================== 1. 信号生成引擎 (增强物理特性) =====================
def generate_signal(mod_type, num_samples=4096, sps=8):
    """
    sps: Samples per symbol, 增加采样率可以让FSK的圆环和时域波形更平滑
    """
    num_symbols = num_samples // sps
    t = np.linspace(0, 1, num_samples)
    iq = np.zeros(num_samples, dtype=complex)

    # a) PSK子类 (采用符号扩展使其在加噪前点迹集中)
    if mod_type in ['BPSK', 'QPSK', '8PSK', 'OQPSK', 'PI/4DQPSK']:
        map_dict = {
            'BPSK': [0, np.pi],
            'QPSK': [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            '8PSK': np.linspace(0, 2 * np.pi, 8, endpoint=False),
            'OQPSK': [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4],
            'PI/4DQPSK': np.linspace(0, 2 * np.pi, 8, endpoint=False)
        }
        syms = np.random.choice(map_dict[mod_type], num_symbols)
        iq = np.repeat(np.exp(1j * syms), sps)

    # b) QAM子类
    elif 'QAM' in mod_type:
        n = int(''.join(filter(str.isdigit, mod_type)))
        side = int(np.sqrt(n))
        x = np.linspace(-side + 1, side - 1, side)
        grid_i, grid_q = np.meshgrid(x, x)
        points = (grid_i + 1j * grid_q).flatten()
        syms = np.random.choice(points, num_symbols)
        iq = np.repeat(syms, sps)

    # c) FSK子类 (修正：相位连续累加，使其在星座图呈现圆环)
    elif mod_type in ['2FSK', '4FSK', 'MSK', 'GMSK']:
        freq_map = {'2FSK': [-1, 1], '4FSK': [-3, -1, 1, 3], 'MSK': [-1, 1], 'GMSK': [-1, 1]}
        data = np.random.choice(freq_map[mod_type], num_symbols)
        # 频率偏差
        h = 0.5 if 'MSK' in mod_type else 1.0
        f_dev = (h / sps) * np.repeat(data, sps)
        phase = np.cumsum(2 * np.pi * f_dev)
        iq = np.exp(1j * phase)

    # d) APSK子类
    elif 'APSK' in mod_type:
        r1, r2 = 1.0, 2.6
        p1 = r1 * np.exp(1j * np.linspace(0, 2 * np.pi, 4, endpoint=False))
        p2 = r2 * np.exp(1j * np.linspace(0, 2 * np.pi, 12, endpoint=False))
        points = np.concatenate([p1, p2])
        syms = np.random.choice(points, num_symbols)
        iq = np.repeat(syms, sps)

    # e) 模拟调制
    elif mod_type in ['AM', 'FM', 'CW', 'DSB']:
        if mod_type == 'AM':
            iq = (1.0 + 0.5 * np.sin(2 * np.pi * 5 * t)) * np.exp(1j * 2 * np.pi * 10 * t)
        elif mod_type == 'FM':
            iq = np.exp(1j * (2 * np.pi * 10 * t + 5 * np.sin(2 * np.pi * 2 * t)))
        elif mod_type == 'CW':
            iq = np.exp(1j * 2 * np.pi * 10 * t)
        elif mod_type == 'DSB':
            iq = np.sin(2 * np.pi * 5 * t) * np.exp(1j * 2 * np.pi * 10 * t)

    # 归一化功率
    iq /= np.sqrt(np.mean(np.abs(iq) ** 2))
    return iq


# ===================== 2. 加噪逻辑 (保持一致) =====================
def add_noise(iq, snr_db):
    signal_power = np.mean(np.abs(iq) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)
    noise = np.random.normal(0, noise_std, iq.shape) + 1j * np.random.normal(0, noise_std, iq.shape)
    return iq + noise


# ===================== 3. 增强版绘图函数 =====================
def plot_modulation_analysis(mod_list, snr=20):
    # 调用字体配置函数，确保绘图前生效
    setup_font()

    num_mods = len(mod_list)
    fig, axes = plt.subplots(num_mods, 3, figsize=(10, 2.2 * num_mods))

    # 调整整体间距，更紧凑
    plt.subplots_adjust(hspace=0.5, wspace=0.25)

    for i, mod in enumerate(mod_list):
        iq = generate_signal(mod)
        noisy_iq = add_noise(iq, snr)

        # --- 列 1: 星座图 (增加范围限制，隐藏刻度) ---
        ax_const = axes[i, 0]
        ax_const.scatter(np.real(noisy_iq), np.imag(noisy_iq), s=0.8, alpha=0.4, color='navy')
        ax_const.set_title(f"{mod}", fontsize=28, fontweight='bold', pad=2)
        ax_const.set_xlim([-3.2, 3.2])
        ax_const.set_ylim([-3.2, 3.2])
        ax_const.set_xticks([])
        ax_const.set_yticks([])
        ax_const.set_aspect('equal')

        # --- 列 2: 时域波形 (I路) ---
        ax_time = axes[i, 1]
        ax_time.plot(np.real(noisy_iq[:256]), linewidth=0.7, color='seagreen')
        ax_time.set_title("Time (I)", fontsize=28, pad=2)
        ax_time.set_xticks([])
        ax_time.set_yticks([])

        # --- 列 3: 功率谱密度 (PSD) - 修正横线问题 ---
        ax_psd = axes[i, 2]
        # 使用更大的nperseg提高频率分辨率
        f, Pxx = signal.welch(noisy_iq, fs=1.0, nperseg=512, detrend=False)
        # 修正：使用fftshift对齐频率中心，并确保绘图数据不交叉
        ax_psd.semilogy(fftshift(f), fftshift(Pxx), color='crimson', linewidth=0.9)
        ax_psd.set_title("Spectrum (PSD)", fontsize=28, pad=2)
        ax_psd.set_xticks([])
        ax_psd.set_yticks([])

    plt.suptitle(f"Modulation Feature Analysis (SNR={snr}dB)", fontsize=28, y=0.99)
    # 强制紧凑布局
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    # 选取代表性子类进行展示
    representative_mods = [
        'QPSK', '16QAM', '8PSK',  # 幅度相位类
        '2FSK', '4FSK', 'GMSK',  # 频率类 (现在星座图应呈现圆环)
        '16APSK',  # 混合类
        'AM', 'FM'  # 模拟类
    ]
    plot_modulation_analysis(representative_mods, snr=20)