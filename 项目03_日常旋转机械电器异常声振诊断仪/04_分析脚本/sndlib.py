# -*- coding: utf-8 -*-
"""
sndlib.py —— 声振信号处理公共库（纯标准库实现，零第三方依赖）

提供：WAV / Phyphox CSV 读取、分帧加窗、基 2 FFT、功率谱、Hilbert 包络、
      时域 / 频域 / 频带 / 谐波 / 包络特征提取。

设计约束
    1. 只依赖 Python 标准库（math / wave / struct / csv / os / json / re）；
    2. 输入输出一律使用 list[float]，任何中间量都能单独打印核对；
    3. FFT 为迭代式 radix-2 Cooley-Tukey，帧长必须补零到 2 的幂。

约定
    audio 模态：手机麦克风录音，fs 通常 44100 / 48000 Hz，关注 0–10 kHz
    vib   模态：手机加速度计（线性加速度），fs 通常 100–200 Hz，关注 0–50 Hz
"""

import math
import os
import csv
import struct
import wave

# ---------------------------------------------------------------- 基础工具

def next_pow2(n):
    """返回不小于 n 的最小 2 的幂。"""
    p = 1
    while p < n:
        p <<= 1
    return p


def safe_name(text):
    """把任意字符串清洗成可做文件名的形式。"""
    bad = '\\/:*?"<>|'
    out = ''.join('_' if c in bad else c for c in str(text)).strip()
    return out or 'unnamed'


def fmt_sec(x):
    """秒数格式化为 '12.34'。"""
    return '%.2f' % float(x)


# ---------------------------------------------------------------- 读文件

def read_wav(path):
    """读取 WAV，返回 (fs, 单声道样本 list[float], 归一化到 ±1)。

    支持 8 / 16 / 24 / 32 bit 整数 PCM。多声道取算术平均。
    """
    with wave.open(path, 'rb') as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        fs = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if sw == 1:
        vals = [(b - 128) / 128.0 for b in raw]
    elif sw == 2:
        cnt = len(raw) // 2
        ints = struct.unpack('<%dh' % cnt, raw[:cnt * 2])
        vals = [v / 32768.0 for v in ints]
    elif sw == 3:
        vals = []
        for i in range(0, len(raw) - 2, 3):
            b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 1 << 24
            vals.append(v / 8388608.0)
    elif sw == 4:
        cnt = len(raw) // 4
        ints = struct.unpack('<%di' % cnt, raw[:cnt * 4])
        vals = [v / 2147483648.0 for v in ints]
    else:
        raise ValueError('不支持的位深：%d byte' % sw)

    if nch > 1:
        n = len(vals) // nch
        mono = []
        for i in range(n):
            s = 0.0
            for c in range(nch):
                s += vals[i * nch + c]
            mono.append(s / nch)
        vals = mono
    return fs, vals


def read_phyphox_csv(path, col=1):
    """读取 Phyphox 导出的 CSV，返回 (fs, values)。

    兼容两行表头（列名 + 单位）、逗号/分号分隔、引号包裹的列名。
    fs 由时间列相邻差分的中位数反推。
    第 0 列为时间，col 指定取第几列（1=X, 2=Y, 3=Z, 4=合成幅值）。
    """
    rows = []
    with open(path, 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        delim = ';' if sample.count(';') > sample.count(',') else ','
        for row in csv.reader(f, delimiter=delim):
            if not row:
                continue
            try:
                rows.append([float(c) for c in row[:max(col + 1, 2)]])
            except ValueError:
                continue  # 表头或单位行
    if len(rows) < 4:
        raise ValueError('CSV 有效数据不足：%s' % path)

    times = [r[0] for r in rows]
    dts = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    dts = sorted(d for d in dts if d > 0)
    if not dts:
        raise ValueError('时间列无正增量：%s' % path)
    dt = dts[len(dts) // 2]
    fs = 1.0 / dt if dt > 0 else 0.0

    idx = col if col < len(rows[0]) else len(rows[0]) - 1
    vals = [r[idx] for r in rows]
    return fs, vals


# ---------------------------------------------------------------- 预处理

def remove_dc(x):
    """去直流：减均值。手机加速度计含重力分量时先做此步再分析。"""
    if not x:
        return []
    m = sum(x) / len(x)
    return [v - m for v in x]


def hann(n):
    """周期 Hann 窗，长度 n。"""
    if n <= 1:
        return [1.0]
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)]


def frame_signal(x, win, hop, windowed=True):
    """滑窗切帧，返回 list[list[float]]；末尾不足一帧则丢弃。"""
    if win <= 0 or hop <= 0:
        raise ValueError('win / hop 必须为正')
    w = hann(win) if windowed else None
    out = []
    n = len(x)
    i = 0
    while i + win <= n:
        seg = x[i:i + win]
        if w is not None:
            seg = [seg[k] * w[k] for k in range(win)]
        out.append(seg)
        i += hop
    return out


# ---------------------------------------------------------------- FFT

def fft(a):
    """迭代式 radix-2 FFT，输入 list[complex] 或 list[float]，长度须为 2 的幂。"""
    n = len(a)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError('FFT 长度必须是 2 的幂，当前 %d' % n)
    buf = [complex(v, 0) if not isinstance(v, complex) else v for v in a]

    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            buf[i], buf[j] = buf[j], buf[i]

    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wl = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for start in range(0, n, length):
            w = 1 + 0j
            for k in range(start, start + half):
                u = buf[k]
                v = buf[k + half] * w
                buf[k] = u + v
                buf[k + half] = u - v
                w *= wl
        length <<= 1
    return buf


def ifft(a):
    """逆 FFT（共轭法）。"""
    n = len(a)
    conj = [v.conjugate() for v in a]
    y = fft(conj)
    return [v.conjugate() / n for v in y]


def spectrum(x, sr):
    """单帧幅度谱，返回 (freqs, mags)，长度 n/2+1，n 为补零后的长度。"""
    n = next_pow2(len(x))
    buf = list(x) + [0.0] * (n - len(x))
    X = fft(buf)
    half = n // 2
    freqs = [k * sr / n for k in range(half + 1)]
    mags = [abs(X[k]) for k in range(half + 1)]
    return freqs, mags


def power_spectrum_avg(x, sr, win_sec=0.5, hop_sec=None):
    """Welch 式平均功率谱：分帧 → 各自幅度谱平方 → 按帧平均。"""
    win = next_pow2(int(win_sec * sr))
    hop = win // 2 if hop_sec is None else max(1, int(hop_sec * sr))
    frames = frame_signal(remove_dc(x), win, hop)
    if not frames:
        return spectrum(x, sr)[0], [0.0] * (next_pow2(len(x)) // 2 + 1)
    acc = None
    freqs = None
    for fr in frames:
        freqs, mags = spectrum(fr, sr)
        if acc is None:
            acc = [m * m for m in mags]
        else:
            for k in range(len(acc)):
                acc[k] += mags[k] * mags[k]
    return freqs, [v / len(frames) for v in acc]


def analytic_envelope(x):
    """Hilbert 包络：构造解析信号后取模。"""
    n = len(x)
    if n == 0:
        return []
    N = next_pow2(n)
    X = fft(list(x) + [0.0] * (N - n))
    h = [0.0] * N
    h[0] = 1.0
    if N % 2 == 0:
        h[N // 2] = 1.0
        for k in range(1, N // 2):
            h[k] = 2.0
    else:
        for k in range(1, (N + 1) // 2):
            h[k] = 2.0
    Y = [X[k] * h[k] for k in range(N)]
    y = ifft(Y)
    return [abs(v) for v in y[:n]]


# ---------------------------------------------------------------- 统计量

def _mean(x):
    return sum(x) / len(x) if x else 0.0


def _std(x):
    if len(x) < 2:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def _rms(x):
    if not x:
        return 0.0
    return math.sqrt(sum(v * v for v in x) / len(x))


def _kurtosis(x):
    n = len(x)
    if n < 4:
        return 0.0
    m = _mean(x)
    m2 = sum((v - m) ** 2 for v in x) / n
    if m2 <= 1e-20:
        return 0.0
    m4 = sum((v - m) ** 4 for v in x) / n
    return m4 / (m2 * m2) - 3.0


def _skew(x):
    n = len(x)
    if n < 3:
        return 0.0
    m = _mean(x)
    m2 = sum((v - m) ** 2 for v in x) / n
    if m2 <= 1e-20:
        return 0.0
    m3 = sum((v - m) ** 3 for v in x) / n
    return m3 / (m2 ** 1.5)


# ---------------------------------------------------------------- 频带定义

AUDIO_BANDS = [(0, 125), (125, 250), (250, 500), (500, 1000),
               (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
VIB_BANDS = [(0, 5), (5, 10), (10, 20), (20, 30),
             (30, 40), (40, 50), (50, 80), (80, 120)]

# 各模态的主频搜索区间（基频）：风扇基频 = 转速/60，1000 rpm ≈ 16.7 Hz
# 声学锚定叶片通过频率 BPF（小风扇基频 228 Hz × 5 叶 ≈ 1141 Hz，实测最稳）；
# 低频旋转基频在手机麦克风上几乎测不到，且各段判定不稳，故不作为锚点。
F0_RANGE = {'audio': (350.0, 1600.0), 'vib': (5.0, 80.0)}
ENV_BAND = (0.5, 100.0)   # 包络谱只看 0–100 Hz 的调制频率


def band_ratios(freqs, psd, bands):
    """各频带功率占总功率（0 频点除外）的比例。"""
    total = 0.0
    for k in range(1, len(psd)):
        total += psd[k]
    out = []
    for lo, hi in bands:
        s = 0.0
        for k in range(1, len(psd)):
            f = freqs[k]
            if lo <= f < hi:
                s += psd[k]
        out.append(s / total if total > 0 else 0.0)
    return out


def dominant_freq(freqs, mags, lo, hi):
    """在 [lo, hi] 内找幅度最大的频点，返回 (频率, 峰值幅度)。"""
    best_f, best_m = 0.0, -1.0
    for k in range(1, len(mags)):
        f = freqs[k]
        if lo <= f <= hi and mags[k] > best_m:
            best_f, best_m = f, mags[k]
    return best_f, max(best_m, 0.0)


def harmonic_ratios(freqs, mags, f0, n_harm=3, tol_ratio=0.06):
    """1× / 2× / 3× 谐波邻域能量占全谱能量的比例。

    谐波存在与否是旋转机械诊断的核心：失衡→1× 抬升，松动→2×/3× 抬升，
    轴承冲击→高频宽频带抬升并调制基频（包络谱可见）。
    """
    total = sum(m * m for m in mags[1:]) or 1.0
    out = []
    for h in range(1, n_harm + 1):
        fc = f0 * h
        tol = max(fc * tol_ratio, freqs[1] - freqs[0] if len(freqs) > 1 else 1.0)
        s = 0.0
        for k in range(1, len(mags)):
            if abs(freqs[k] - fc) <= tol:
                s += mags[k] * mags[k]
        out.append(s / total)
    return out


# ---------------------------------------------------------------- 特征提取

FEATURE_NAMES = [
    # 时域 9
    'rms', 'peak', 'p2p', 'crest', 'kurtosis', 'skew', 'zcr', 'impulse', 'clearance',
    # 频域 8
    'spec_centroid', 'spec_bandwidth', 'spec_rolloff85', 'spec_flatness',
    'spec_entropy', 'spec_flux', 'spec_slope', 'dom_freq',
    # 频带 8
    'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7',
    # 谐波 3 + 基频 1
    'f0', 'h1_ratio', 'h2_ratio', 'h3_ratio',
    # 包络 3
    'env_rms', 'env_crest', 'env_dom_freq',
]


def _zcr(x):
    if len(x) < 2:
        return 0.0
    c = 0
    for i in range(1, len(x)):
        if (x[i - 1] >= 0) != (x[i] >= 0):
            c += 1
    return c / (len(x) - 1)


def window_features(frame, sr, modality='audio', prev_mag=None):
    """对一帧（已加窗）提取全部特征，返回 dict。"""
    x = remove_dc(frame)
    n = len(x)
    feats = {}

    # ---- 时域
    peak = max(abs(v) for v in x) if n else 0.0
    rms = _rms(x)
    feats['rms'] = rms
    feats['peak'] = peak
    feats['p2p'] = (max(x) - min(x)) if n else 0.0
    feats['crest'] = peak / rms if rms > 1e-12 else 0.0
    feats['kurtosis'] = _kurtosis(x)
    feats['skew'] = _skew(x)
    feats['zcr'] = _zcr(x)
    feats['impulse'] = peak / (sum(abs(v) for v in x) / n) if n and sum(abs(v) for v in x) > 1e-12 else 0.0
    mean_sqrt_amp = (sum(math.sqrt(abs(v)) for v in x) / n) ** 2 if n else 0.0
    feats['clearance'] = peak / mean_sqrt_amp if mean_sqrt_amp > 1e-12 else 0.0

    # ---- 频域
    freqs, mags = spectrum(x, sr)
    psd = [m * m for m in mags]
    total = sum(psd[1:]) or 1e-30
    centroid = sum(freqs[k] * psd[k] for k in range(1, len(psd))) / total
    feats['spec_centroid'] = centroid
    feats['spec_bandwidth'] = math.sqrt(
        sum(((freqs[k] - centroid) ** 2) * psd[k] for k in range(1, len(psd))) / total)
    cum, rolloff, target = 0.0, 0.0, 0.85 * total
    for k in range(1, len(psd)):
        cum += psd[k]
        if cum >= target:
            rolloff = freqs[k]
            break
    feats['spec_rolloff85'] = rolloff
    nz = [p for p in psd[1:] if p > 1e-30]
    if nz:
        gm = math.exp(sum(math.log(p) for p in nz) / len(nz))
        am = sum(nz) / len(nz)
        feats['spec_flatness'] = gm / am if am > 0 else 0.0
    else:
        feats['spec_flatness'] = 0.0
    # 谱熵（归一化到 0–1）
    ent = 0.0
    for k in range(1, len(psd)):
        p = psd[k] / total
        if p > 1e-30:
            ent -= p * math.log(p)
    feats['spec_entropy'] = ent / math.log(len(psd) - 1) if len(psd) > 2 else 0.0
    # 谱通量（相邻帧归一化幅度谱的欧氏距离）
    if prev_mag is None:
        feats['spec_flux'] = 0.0
    else:
        s = sum(mags[1:]) or 1.0
        sp = sum(prev_mag[1:]) or 1.0
        feats['spec_flux'] = math.sqrt(
            sum((mags[k] / s - prev_mag[k] / sp) ** 2 for k in range(1, len(mags))))
    # 谱斜率（线性拟合 PSD 对频率的斜率，反映高频衰减快慢）
    m = len(freqs) - 1
    if m > 2:
        fx = freqs[1:]
        fy = [10 * math.log10(psd[k] + 1e-30) for k in range(1, len(psd))]
        mx, my = sum(fx) / m, sum(fy) / m
        num = sum((fx[i] - mx) * (fy[i] - my) for i in range(m))
        den = sum((fx[i] - mx) ** 2 for i in range(m)) or 1e-30
        feats['spec_slope'] = num / den
    else:
        feats['spec_slope'] = 0.0

    lo, hi = F0_RANGE.get(modality, (5.0, 600.0))
    f0, _ = dominant_freq(freqs, mags, lo, hi)
    feats['dom_freq'] = f0
    feats['f0'] = f0

    bands = AUDIO_BANDS if modality == 'audio' else VIB_BANDS
    br = band_ratios(freqs, psd, bands)
    for i, v in enumerate(br):
        feats['b%d' % i] = v

    hr = harmonic_ratios(freqs, mags, f0 if f0 > 0 else 1.0)
    feats['h1_ratio'], feats['h2_ratio'], feats['h3_ratio'] = hr[0], hr[1], hr[2]

    # ---- 包络（冲击/调制检测）
    # 包络的带宽远低于原信号，先 4 倍抽取再算 Hilbert，速度提升约 16 倍且不影响结论
    dec = 4 if sr > 8000 else 1
    env = analytic_envelope(x[::dec])
    feats['env_rms'] = _rms(env)
    epeak = max(env) if env else 0.0
    feats['env_crest'] = epeak / feats['env_rms'] if feats['env_rms'] > 1e-12 else 0.0
    efreqs, emags = spectrum([v - _mean(env) for v in env], sr / float(dec))
    e_f0, _ = dominant_freq(efreqs, emags, ENV_BAND[0], ENV_BAND[1])
    feats['env_dom_freq'] = e_f0

    return feats, mags


def extract_windows(x, sr, modality='audio', win_sec=0.5, hop_ratio=0.5):
    """整段信号 → 逐窗特征列表，返回 list[(t_start, dict)]。"""
    win = next_pow2(int(win_sec * sr))
    hop = max(1, int(win * hop_ratio))
    frames = frame_signal(remove_dc(x), win, hop)
    out = []
    prev = None
    for i, fr in enumerate(frames):
        feats, mags = window_features(fr, sr, modality, prev)
        out.append((i * hop / float(sr), feats))
        prev = mags
    return out


# ---------------------------------------------------------------- 杂项

def file_size_kb(path):
    try:
        return os.path.getsize(path) / 1024.0
    except OSError:
        return 0.0


def db(x, floor=-120.0):
    """线性幅度转 dB。"""
    if x <= 0:
        return floor
    return max(floor, 20.0 * math.log10(x))
