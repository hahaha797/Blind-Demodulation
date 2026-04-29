import numpy as np
import matplotlib.pyplot as plt

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
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['xtick.labelsize'] = 26
    plt.rcParams['ytick.labelsize'] = 26
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['figure.titlesize'] = 22

    print("✅ Font initialized: English=Times New Roman, Chinese=SimSun")

# ===================== 1. 模拟信号生成函数 =====================
def generate_baseband_signal(mod_type, num_symbols=10000):
    """简单模拟生成各调制方式的基带IQ点"""
    t = np.linspace(0, 1, num_symbols)

    if 'QAM' in mod_type:
        n = int(mod_type.replace('QAM', ''))
        side = int(np.sqrt(n))
        x = np.linspace(-side + 1, side - 1, side)
        i, q = np.meshgrid(x, x)
        points = (i + 1j * q).flatten()
        indices = np.random.choice(len(points), num_symbols)
        iq = points[indices]

    elif 'PSK' in mod_type:
        if 'BPSK' in mod_type:
            phases = [0, np.pi]
        elif 'QPSK' in mod_type:
            phases = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        elif '8PSK' in mod_type:
            phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        else:
            phases = np.linspace(0, 2 * np.pi, 4, endpoint=False)  # 默认
        iq = np.exp(1j * np.random.choice(phases, num_symbols))

    elif 'FSK' in mod_type or 'MSK' in mod_type or 'GMSK' in mod_type:
        # FSK在星座图上通常表现为圆环（包络恒定，相位连续变化）
        freq = 5 if '4FSK' in mod_type else 2
        iq = np.exp(1j * (2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi)))

    elif 'APSK' in mod_type:
        # 16APSK 常用半径比例
        r1, r2 = 1.0, 2.5
        theta1 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        theta2 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        p1 = r1 * np.exp(1j * theta1)
        p2 = r2 * np.exp(1j * theta2)
        points = np.concatenate([p1, p2])
        iq = np.random.choice(points, num_symbols)

    else:  # 模拟调制 (AM, FM, DSB, CW)
        # 模拟调制在IQ域通常表现为特定轨迹
        if mod_type == 'AM':
            iq = (1.5 + np.sin(2 * np.pi * 2 * t)) * np.exp(1j * 0)
        elif mod_type == 'FM':
            iq = np.exp(1j * np.cumsum(np.sin(2 * np.pi * 2 * t)))
        else:
            iq = (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)) * 0.5

    # 归一化功率
    iq = iq / np.sqrt(np.mean(np.abs(iq) ** 2))
    return np.vstack([np.real(iq), np.imag(iq)])

# ===================== 2. 加噪函数 =====================
def add_awgn(signal, snr_db):
    signal_power = np.mean(np.sum(signal ** 2, axis=0))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2)
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise

# ===================== 3. 绘图主程序 =====================
def plot_modulation_constellations(snr_list=[30, 15, 5]):
    # 🔴 在这里调用字体设置，确保绘图前生效
    setup_font()

    mod_list = [
        'BPSK', 'QPSK', '8PSK', 'OQPSK', 'PI/4DQPSK',
        '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
        '2FSK', '4FSK', 'MSK', 'GMSK',
        '16APSK', '32APSK',
        'AM', 'FM', 'DSB', 'CW'
    ]

    num_mods = len(mod_list)
    num_snrs = len(snr_list)

    fig, axes = plt.subplots(num_mods, num_snrs, figsize=(4 * num_snrs, 3 * num_mods))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    for m_idx, mod_type in enumerate(mod_list):
        # 生成纯净信号
        raw_signal = generate_baseband_signal(mod_type)

        for s_idx, snr in enumerate(snr_list):
            # 添加噪声
            noisy_signal = add_awgn(raw_signal, snr)

            ax = axes[m_idx, s_idx]
            ax.scatter(noisy_signal[0], noisy_signal[1], s=1, alpha=0.6, color='blue')
            ax.set_title(f"{mod_type} @ {snr}dB", fontsize=26)
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_aspect('equal')

            if m_idx == num_mods - 1:
                ax.set_xlabel("In-Phase")
            if s_idx == 0:
                ax.set_ylabel("Quadrature")

    plt.suptitle("", fontsize=16, y=0.99)
    # 调整布局
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

if __name__ == "__main__":
    plot_modulation_constellations(snr_list=[30, 20, 10, 5])