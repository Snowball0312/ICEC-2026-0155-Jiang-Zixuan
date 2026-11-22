# -*- coding: utf-8 -*-
"""
selftest.py —— 管线自检：不碰真实数据，用合成信号把「读音频 → 提特征 → 训练 → 盲测」跑一遍

用法
    python selftest.py            # 单元校验 + 端到端自检，约 1 分钟
    python selftest.py --quick    # 只跑单元校验，几秒钟

重要说明
    本脚本生成的 WAV 是**数学合成信号**，只用来验证代码逻辑正确，
    绝不可当作实验数据提交，也不放进 01_原始声振数据。
    输出目录 `_selftest_tmp/` 已在 .gitignore 中排除。

自检内容
    1. FFT 与朴素 DFT 逐点对比（误差 < 1e-9）
    2. ifft(fft(x)) == x
    3. 1000 Hz 正弦的谱峰定位误差
    4. AM 信号（载波 1500 Hz / 调制 20 Hz）的包络谱峰定位
    5. 合成 5 类工况 × 6 轮（2 档 × 3 轮）→ 提特征 → 训练 → 交叉验证准确率
"""

import os
import sys
import csv
import math
import wave
import shutil
import random
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, '_selftest_tmp')

CLASSES = [('L0', '正常'), ('L1', '进风半挡'), ('L2', '进风强挡'),
           ('L3', '出风半挡'), ('L4', '出风强挡')]

# 合成信号用的物理参数（模拟手持小风扇：4500 rpm、5 叶、基频 75 Hz）
F0_HZ = 110.0
BPF_H = 5


def naive_dft(a):
    n = len(a)
    out = []
    for k in range(n):
        s = 0j
        for t in range(n):
            ang = -2.0 * math.pi * k * t / n
            s += a[t] * complex(math.cos(ang), math.sin(ang))
        out.append(s)
    return out


def check_units():
    ok = True

    # 1. FFT vs DFT
    rng = random.Random(7)
    x = [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(64)]
    a = sndlib.fft(x)
    b = naive_dft(x)
    err = max(abs(a[i] - b[i]) for i in range(64))
    print('  FFT 与朴素 DFT 最大偏差 %.3e  %s' % (err, '通过' if err < 1e-9 else '失败'))
    ok &= err < 1e-9

    # 2. ifft 往返
    y = sndlib.ifft(sndlib.fft(x))
    err2 = max(abs(y[i] - x[i]) for i in range(64))
    print('  ifft(fft(x)) 与 x 最大偏差 %.3e  %s' % (err2, '通过' if err2 < 1e-9 else '失败'))
    ok &= err2 < 1e-9

    # 3. 正弦谱峰
    fs = 8000
    sig = [math.sin(2 * math.pi * 1000 * i / fs) for i in range(fs)]
    freqs, mags = sndlib.spectrum(sig, fs)
    f_peak, _ = sndlib.dominant_freq(freqs, mags, 100.0, 3000.0)
    print('  1000 Hz 正弦谱峰 %.1f Hz  %s' % (f_peak, '通过' if abs(f_peak - 1000) < 5 else '失败'))
    ok &= abs(f_peak - 1000) < 5

    # 4. AM 包络
    am = [(1 + 0.7 * math.sin(2 * math.pi * 20 * i / fs)) * math.sin(2 * math.pi * 1500 * i / fs)
          for i in range(fs)]
    env = sndlib.analytic_envelope(am)
    efreqs, emags = sndlib.spectrum([v - sum(env) / len(env) for v in env], fs)
    e_f, _ = sndlib.dominant_freq(efreqs, emags, 5.0, 200.0)
    print('  20 Hz 调制包络谱峰 %.1f Hz  %s' % (e_f, '通过' if abs(e_f - 20) < 3 else '失败'))
    ok &= abs(e_f - 20) < 3
    return ok


# ---------------------------------------------------------------- 合成信号

def synth(cls, dur=3.0, fs=16000, seed=0, gear=2):
    """按工况机理合成一段可区分的声音（仅用于自检，不是实验数据）。

    机理对应关系（与 01_原始声振数据/00_工况定义与采集方案.md 一致）：
        L0 正常        基频 + 谐波 + 低宽带
        L1 进风半挡    谐波略降 + 轻度高频滚降 + 宽带上升
        L2 进风强挡    基频下移 + 6 Hz 喘振调制 + 强高频滚降
        L3 出风半挡    BPF 被基频调制 → 边带 + 宽带抬升
        L4 出风强挡    转速略升 + BPF/谐波变尖 + 低频轰鸣 + 宽带抬升

    gear 模拟三档转速（约 ±4%），用来检验留一档位验证这条路径能跑通。
    """
    rng = random.Random(seed)
    n = int(dur * fs)
    speed = {1: 0.96, 2: 1.00, 3: 1.04}.get(gear, 1.00)
    detune = {'L0': 1.00, 'L1': 0.985, 'L2': 0.93, 'L3': 0.99, 'L4': 1.03}[cls]
    f0 = F0_HZ * detune * speed * (1 + rng.uniform(-0.01, 0.01))
    tone = [0.0] * n

    def add(f, amp, ph=0.0):
        for i in range(n):
            tone[i] += amp * math.sin(2 * math.pi * f * i / fs + ph)

    if cls == 'L0':                                    # 基准
        for k in range(1, 9):
            add(f0 * k, 0.30 / k)
        broadband = 0.030
    elif cls == 'L1':                                  # 轻度节流：谐波在、高频掉
        for k in range(1, 9):
            add(f0 * k, 0.28 / k)
        broadband = 0.055
    elif cls == 'L2':                                  # 重度节流：低频涡流 + 喘振
        for k in range(1, 6):
            add(f0 * k, 0.24 / k)
        broadband = 0.10
    elif cls == 'L3':                                  # 背压：BPF 被基频调制
        bpf = BPF_H * f0
        for k in range(1, 9):
            add(f0 * k, 0.55 if k == BPF_H else 0.22 / k)
        for i in range(n):
            tone[i] *= 1.0 + 0.60 * math.sin(2 * math.pi * f0 * i / fs)
        broadband = 0.065
    else:                                              # 出风强挡：强背压
        bpf = BPF_H * f0
        for k in range(1, 10):
            add(f0 * k, 0.62 if k == BPF_H else 0.26 / k)
        for f in (25.0, 40.0, 55.0):                   # 背压涡流的低频轰鸣
            add(f, 0.16)
        for i in range(n):
            tone[i] *= 1.0 + 0.70 * math.sin(2 * math.pi * f0 * i / fs)
        broadband = 0.140

    sig = tone
    if cls == 'L2':                                    # 数 Hz 喘振调制
        for i in range(n):
            sig[i] *= 1.0 + 0.35 * math.sin(2 * math.pi * 6.0 * i / fs)

    noise_gain = rng.uniform(0.85, 1.15)               # 每轮环境噪声略有差别（安静室内 ±15%）
    for i in range(n):
        sig[i] += broadband * noise_gain * rng.gauss(0, 1)

    if cls in ('L1', 'L2'):                            # 进气受阻的高频衰减
        alpha, gain = (0.45, 1.35) if cls == 'L1' else (0.12, 2.7)
        y = 0.0
        for i in range(n):
            y += alpha * (sig[i] - y)
            sig[i] = y * gain

    peak = max(abs(v) for v in sig) or 1.0
    return [0.85 * v / peak for v in sig]


def write_wav(path, fs, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(b''.join(int(max(-1.0, min(1.0, v)) * 32767).to_bytes(2, 'little', signed=True)
                               for v in x))


GEARS = (1, 1, 1, 2, 2, 2)   # 与采集手册的 6 轮表一致：1 档 3 轮 + 2 档 3 轮


def round_meta(r):
    """合成数据的轮次元信息：距离固定 12 cm，档位按手册表轮换（三档各 1 轮）。"""
    return GEARS[(r - 1) % len(GEARS)], 12


def build_dataset():
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP, exist_ok=True)
    fs = 16000
    for r in range(1, 7):
        gear, dist = round_meta(r)
        for code, name in CLASSES:
            wav = os.path.join(TMP, '%s_r%02d_g%d_%dcm_1121_1500.wav' % (code, r, gear, dist))
            write_wav(wav, fs, synth(code, seed=r * 10 + ord(code[1]), fs=fs, gear=gear))
    return fs


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit('子流程失败：%s' % ' '.join(cmd))
    return r.stdout


def check_pipeline():
    fs = build_dataset()
    feat_csv = os.path.join(TMP, 'features.csv')
    print('\n  合成 5 类 × 6 轮 = 30 段 WAV 至 _selftest_tmp/（%d Hz）' % fs)

    out = run(['features.py', '--root', TMP, '--out', feat_csv])
    tail = [l for l in out.strip().splitlines() if '特征行' in l]
    print('  %s' % (tail[0] if tail else '特征提取完成'))

    out = run(['train.py', '--features', feat_csv, '--outdir', TMP])
    acc_line = [l for l in out.splitlines() if '窗口级准确率' in l]
    print('  %s' % (acc_line[0] if acc_line else '训练完成'))

    with open(feat_csv, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    gear, dist = round_meta(15)
    test_wav = os.path.join(TMP, 'L1_r03_g%d_%dcm_1121_1500.wav' % (gear, dist))
    out = run(['predict.py', test_wav, '--model', os.path.join(TMP, 'model.json')])
    hit = '诊断结论　：L1' in out
    print('  盲测单段（L1 进风半挡）：%s  %s' % ('判定为 L1' if hit else '判定错误',
                                          '通过' if hit else '失败'))
    acc_ok = True
    if acc_line:
        try:
            acc = float(acc_line[0].split('窗口级准确率')[1].split()[0])
            acc_ok = acc >= 0.85
        except (IndexError, ValueError):
            acc_ok = False
    print('  端到端准确率：%s' % ('通过' if acc_ok else '偏低（合成信号下应 > 85%）'))
    return acc_ok and hit


def main():
    ap = argparse.ArgumentParser(description='管线自检')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--keep', action='store_true', help='保留 _selftest_tmp 便于排查')
    args = ap.parse_args()

    print('一、单元校验')
    ok1 = check_units()
    ok2 = True
    if not args.quick:
        print('\n二、端到端自检')
        ok2 = check_pipeline()
        if not args.keep and os.path.isdir(TMP):
            shutil.rmtree(TMP)

    print('\n结论：%s' % ('全部通过，管线可用' if (ok1 and ok2) else '存在失败项，请看上方标注'))
    return 0 if (ok1 and ok2) else 1


if __name__ == '__main__':
    sys.exit(main())
