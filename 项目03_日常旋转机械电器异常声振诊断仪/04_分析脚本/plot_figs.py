# -*- coding: utf-8 -*-
"""
plot_figs.py —— 一键出图：把原始数据与结果表变成 02_可视化图 里的 6 张图（SVG，可转 PNG）

用法
    python plot_figs.py                    # 能出几张出几张
    python plot_figs.py --only 1,2,3       # 只出前三张
    python plot_figs.py --pick 1           # 每工况取第 2 段录音画波形/频谱

图目录
    图1_五工况波形对比      每种工况 2 s 时域波形，各自峰值归一化，看幅值包络与冲击
    图2_五工况频谱对比      五种工况的平均功率谱叠在一张（0–4 kHz），看各工况改变的是哪一段
    图3_包络谱对比          正常 vs 四种遮挡的 Hilbert 包络谱（0–100 Hz），看喘振与边带
    图4_特征散点            两个区分度最高的特征，按工况着色，论证线性可分
    图5_混淆矩阵            读 混淆矩阵_窗口.csv，画热力格
    图6_抗噪曲线            读 抗噪曲线.csv，准确率 vs 信噪比

缺数据时对应图自动跳过并提示缺哪个文件，不会中途崩。
"""

import os
import csv
import sys
import math
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib
import plotsvg
from plotsvg import Chart, COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PROJ, '01_原始声振数据')
MODEL_DIR = os.path.join(PROJ, '03_特征与模型')
FIG_DIR = os.path.join(PROJ, '02_可视化图')

CODES = ['L0', 'L1', 'L2', 'L3', 'L4']
FEAT_CN = {'spec_centroid': '谱质心 / Hz', 'kurtosis': '峭度', 'crest': '波峰因子', 'rms': 'RMS',
           'skew': '偏度', 'zcr': '过零率', 'spec_entropy': '谱熵', 'spec_flatness': '谱平坦度',
           'spec_bandwidth': '谱带宽 / Hz', 'dom_freq': '主频 / Hz', 'env_rms': '包络 RMS',
           'env_crest': '包络波峰因子', 'h1_ratio': '1× 谐波占比', 'h2_ratio': '2× 谐波占比',
           'impulse': '脉冲因子', 'clearance': '裕度因子'}
NAMES = {'L0': '正常', 'L1': '一档半遮挡', 'L2': '一档全遮挡',
         'L3': '二档半遮挡', 'L4': '二档全遮挡'}
PICK = 0


def welch_avg(x, fs, seg=1024, overlap=512):
    """Welch 平均功率谱：分段加 Hann 窗、功率谱平均后再开方，返回 (freqs, mags)。"""
    x = [v - sndlib._mean(x) for v in x]
    win = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (seg - 1)) for i in range(seg)]
    acc, cnt, i = None, 0, 0
    while i + seg <= len(x):
        f, m = sndlib.spectrum([x[i + k] * win[k] for k in range(seg)], fs)
        p = [v * v for v in m]
        acc = p if acc is None else [a + b for a, b in zip(acc, p)]
        cnt += 1
        i += seg - overlap
    if not cnt:
        return sndlib.spectrum(x, fs)
    return f, [(v / cnt) ** 0.5 for v in acc]


def cond_wavs():
    """返回 {工况码: [wav 路径...]}。"""
    out = {}
    for code in CODES:
        for d in sorted(os.listdir(DATA)):
            full = os.path.join(DATA, d)
            if not os.path.isdir(full) or not d.startswith(code):
                continue
            files = sorted(f for f in os.listdir(full) if f.lower().endswith('.wav'))
            if files:
                out[code] = [os.path.join(full, f) for f in files]
    return out


def pick_wav(mapping, code):
    lst = mapping.get(code) or []
    return lst[min(PICK, len(lst) - 1)] if lst else None


def downsample(freqs, vals, fmax, max_pts=420):
    """把谱线裁到 fmax 并降采样，避免 SVG 里几万个点。"""
    sel = [(f, v) for f, v in zip(freqs, vals) if f <= fmax]
    if not sel:
        return [], []
    step = max(1, len(sel) // max_pts)
    sel = sel[::step]
    return [s[0] for s in sel], [s[1] for s in sel]


# ---------------------------------------------------------------- 图 1

def fig1(mapping, outdir):
    rows = [(code, pick_wav(mapping, code)) for code in CODES]
    rows = [(c, p) for c, p in rows if p]
    if not rows:
        print('  图1 跳过：01_原始声振数据 下没有 WAV')
        return None

    amp = 0.34
    ch = Chart(960, 92 * len(rows) + 152, title='图 1　五工况时域波形对比（各取前 2 s，峰值归一化）',
               xlabel='时间 / s', xlim=(0, 2), ylim=(-0.5, len(rows) - 0.5), pad=(78, 34, 60, 64))
    ch.axes(yticks='none')
    for i, (code, path) in enumerate(rows):
        sr, x = sndlib.read_wav(path)
        seg = x[:int(2 * sr)]
        peak = max((abs(v) for v in seg), default=1.0) or 1.0
        base = len(rows) - 1 - i
        if i:
            ch.body.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="0.5"/>'
                           % (ch.pl, ch.py(base + 0.5), ch.w - ch.pr, ch.py(base + 0.5), plotsvg.GRID))
        ys = [v / peak * amp + base for v in seg]
        xs = [k / sr for k in range(len(seg))]
        step = max(1, len(xs) // 1600)
        ch.line(xs[::step], ys[::step], COLORS[i % 5], 1.0)
        rms = sndlib._rms(sndlib.remove_dc(seg))
        cy = ch.py(base + 0.40)
        ch.body.append('<text x="%d" y="%.1f" font-size="12.5" fill="%s" dominant-baseline="central">'
                       '%s %s</text>' % (ch.pl + 6, cy, plotsvg.INK2, code, NAMES[code]))
        ch.body.append('<text x="%d" y="%.1f" font-size="11" fill="%s" text-anchor="end" '
                       'dominant-baseline="central">RMS %.3f</text>'
                       % (ch.w - ch.pr, cy, plotsvg.INK3, rms))
    ch.note('每行独立峰值归一化，比较的是波形形状与包络，不是绝对音量；右端 RMS 为该段原始有效值')
    return ch.save(os.path.join(outdir, '图1_五工况波形对比.svg'))


# ---------------------------------------------------------------- 图 2

def fig2(mapping, outdir):
    ch = Chart(960, 540, title='图 2　五工况平均功率谱对比（0–4 kHz，各自峰值归一化）',
               xlabel='频率 / Hz', ylabel='相对幅值 / dB', xlim=(0, 4000), ylim=(-60, 3),
               pad=(78, 34, 54, 76))
    ch.axes()
    drew = 0
    for i, code in enumerate(CODES):
        path = pick_wav(mapping, code)
        if not path:
            continue
        sr, x = sndlib.read_wav(path)
        freqs, psd = sndlib.power_spectrum_avg(x, sr, 0.5)
        if not psd:
            continue
        mx = max(psd) or 1.0
        db = [10 * math.log10(v / mx + 1e-12) for v in psd]
        fx, fy = downsample(freqs, db, 4000)
        ch.line(fx, fy, COLORS[i % 5], 1.4, label='%s %s' % (code, NAMES[code]))
        drew += 1
    if not drew:
        print('  图2 跳过：没有可用的 WAV')
        return None
    ch.legend(x=ch.w - ch.pr - 132, y=ch.pt + 10)
    ch.note('遮挡方式决定转速与谱形：半遮挡 661–749 Hz（纸片贴叶制动、平坦度低）、全遮挡 1074–1386 Hz（近乎空载、宽带成分高）；同一遮挡状态下换档只差 10–12%')
    return ch.save(os.path.join(outdir, '图2_五工况频谱对比.svg'))


# ---------------------------------------------------------------- 图 3

def fig3(mapping, outdir):
    ch = Chart(960, 500, title='图 3　Hilbert 包络谱对比（0–100 Hz）', xlabel='包络频率 / Hz',
               ylabel='相对幅值 / dB', xlim=(0, 100), ylim=(-30, 3), pad=(78, 34, 54, 78))
    ch.axes()
    drew = 0
    for i, code in enumerate(CODES):
        if code not in ('L0', 'L2', 'L3', 'L4'):   # L1 与 L3 同属半遮挡、转速只差 11%，谱形几乎重合，只画 L3
            continue
        path = pick_wav(mapping, code)
        if not path:
            continue
        sr, x = sndlib.read_wav(path)
        dec = 4 if sr > 8000 else 1
        env = sndlib.analytic_envelope(x[::dec])
        # 包络谱只关心 0–100 Hz，先把包络做块平均降到约 250 Hz 再 FFT。
        # 直接用 12 kHz 的 262144 点 FFT（纯 Python 无 numpy）要 5 分钟/文件，
        # 降采样后 1 秒内完成，谱形肉眼无差别（奈奎斯特 125 Hz，覆盖 0–100 Hz）。
        step = max(1, int(round((sr / float(dec)) / 250.0)))
        env = [sum(env[i:i + step]) / step
               for i in range(0, len(env) - step + 1, step)]
        fs_env = (sr / float(dec)) / step
        # Welch 平均：单段 0.5 s 的包络谱起伏极大（谱线全是毛刺），
        # 取 4 s 段、50% 重叠、功率谱平均后再开方，才能看出"包络在哪个频率上调制"。
        efreqs, emags = welch_avg(env, fs_env, seg=512, overlap=384)
        emx = max(emags) or 1.0
        db = [20 * math.log10(v / emx + 1e-12) for v in emags]
        fx, fy = downsample(efreqs, db, 100, 300)
        ch.line(fx, fy, COLORS[i % 5], 1.4, label='%s %s' % (code, NAMES[code]))
        drew += 1
    if not drew:
        print('  图3 跳过：没有可用的 WAV')
        return None
    ch.legend(x=ch.pl + 300, y=ch.py(-22))
    ch.note('包络电平不随遮挡程度单调变化（L4 最高、L2 最低），它是转速与遮挡的共同结果；可用信息在谱线位置而非整体电平。0–3 Hz 的抬升来自转速慢变与拍频，属包络近直流成分，不作判据')
    return ch.save(os.path.join(outdir, '图3_包络谱对比.svg'))


# ---------------------------------------------------------------- 图 4

def fig4(outdir, xkey='spec_centroid', ykey='kurtosis'):
    path = os.path.join(MODEL_DIR, 'features.csv')
    if not os.path.isfile(path):
        print('  图4 跳过：缺少 %s' % path)
        return None
    rows = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['modality'] != 'audio':
                continue
            try:
                rows.append((r['condition'], float(r[xkey]), float(r[ykey])))
            except (KeyError, ValueError):
                continue
    if not rows:
        print('  图4 跳过：features.csv 里没有 audio 行')
        return None
    xs = [v for _c, v, _y in rows]
    ys = [v for _c, _x, v in rows]
    xlim = (min(xs) * 0.98, max(xs) * 1.02)
    ylim = (min(ys) * 0.98, max(ys) * 1.02)
    ch = Chart(960, 520, title='图 4　特征空间分布：%s × %s' % (FEAT_CN.get(xkey, xkey), FEAT_CN.get(ykey, ykey)),
               xlabel=FEAT_CN.get(xkey, xkey), ylabel=FEAT_CN.get(ykey, ykey), xlim=xlim, ylim=ylim)
    ch.axes()
    for i, code in enumerate(CODES):
        pts = [(x, y) for c, x, y in rows if c == code]
        if not pts:
            continue
        for x, y in pts:
            ch.body.append('<circle cx="%.1f" cy="%.1f" r="3.1" fill="%s" opacity="0.62"/>'
                           % (ch.px(x), ch.py(y), COLORS[i % 5]))
        ch.legends.append(('%s %s' % (code, NAMES[code]), COLORS[i % 5]))
    ch.legend(x=ch.w - ch.pr - 140, y=ch.pt + 10)
    ch.note('每个点是一个 0.5 s 分析窗；类间重叠越小，线性分类器越够用')
    return ch.save(os.path.join(outdir, '图4_特征散点.svg'))


# ---------------------------------------------------------------- 图 5

def fig5(outdir):
    path = os.path.join(MODEL_DIR, '混淆矩阵_窗口.csv')
    if not os.path.isfile(path):
        print('  图5 跳过：缺少 %s' % path)
        return None
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    labels = [x.strip() for x in rows[0][1:] if x.strip()]
    mat = []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        mat.append([int(float(v)) for v in r[1:1 + len(labels)]])
    if not mat:
        print('  图5 跳过：混淆矩阵为空')
        return None
    ch = Chart(760, 620, title='图 5　混淆矩阵（窗口级，按轮次 GroupKFold 汇总）',
               pad=(152, 40, 56, 112), xlim=(0, 1), ylim=(0, 1))
    ch.heat(mat, ['%s %s' % (c, NAMES.get(c, '')) for c in [r[0] for r in rows[1:] if r and r[0].strip()]],
            ['%s %s' % (l, NAMES.get(l, '')) for l in labels],
            vmax=max(max(r) for r in mat))
    ch.body.append('<text x="%.1f" y="%d" font-size="12.5" fill="%s" text-anchor="middle">预测工况</text>'
                   % ((ch.w) / 2, ch.h - 8, plotsvg.INK2))
    ch.body.append('<text x="24" y="%.1f" font-size="12.5" fill="%s" text-anchor="middle" '
                   'transform="rotate(-90 24 %.1f)">真实工况</text>'
                   % ((ch.pt + ch.h - ch.pb) / 2, plotsvg.INK2, (ch.pt + ch.h - ch.pb) / 2))
    return ch.save(os.path.join(outdir, '图5_混淆矩阵.svg'))


# ---------------------------------------------------------------- 图 6

def fig6(outdir):
    path = os.path.join(MODEL_DIR, '抗噪曲线.csv')
    if not os.path.isfile(path):
        print('  图6 跳过：缺少 %s（先跑 robustness.py）' % path)
        return None
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print('  图6 跳过：抗噪曲线为空')
        return None
    order = []
    for r in rows:
        tag = r['信噪比_dB']
        order.append((99.0 if tag in ('无噪声', 'clean') else float(tag), tag, float(r['整段准确率'])))
    order.sort()
    xs = list(range(len(order)))
    ys = [o[2] * 100 for o in order]
    ch = Chart(880, 480, title='图 6　抗噪能力：整段准确率 vs 信噪比',
               xlabel='信噪比（越靠右噪声越强）', ylabel='整段准确率 / %',
               xlim=(-0.5, len(xs) - 0.15), ylim=(0, 112), pad=(78, 34, 54, 96))
    ch.axes(xticks='none')
    ch.area(xs, ys, COLORS[0], 0.12)
    ch.line(xs, ys, COLORS[0], 2.0)
    for x, y in zip(xs, ys):
        ch.body.append('<circle cx="%.1f" cy="%.1f" r="3.4" fill="%s"/>' % (ch.px(x), ch.py(y), COLORS[0]))
        ch.body.append('<text x="%.1f" y="%.1f" font-size="11.5" fill="%s" text-anchor="middle">%.1f</text>'
                       % (ch.px(x), ch.py(y) - 11, plotsvg.INK2, y))
    for x, (v, tag, _a) in zip(xs, order):
        ch.body.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="middle">%s</text>'
                       % (ch.px(x), ch.h - ch.pb + 19, plotsvg.INK3, tag))
    ch.note('噪声按信号 RMS 定标叠加；横轴为信噪比档位，从左到右噪声逐渐增强')
    return ch.save(os.path.join(outdir, '图6_抗噪曲线.svg'))


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description='一键出图')
    ap.add_argument('--only', default='1,2,3,4,5,6')
    ap.add_argument('--pick', type=int, default=0, help='每工况取第几段录音（从 0 计）')
    ap.add_argument('--png', action='store_true', help='同时调用 svg_to_png.py 转 PNG')
    ap.add_argument('--data', default=None, help='原始数据目录（默认 01_原始声振数据）')
    ap.add_argument('--modeldir', default=None, help='结果目录（默认 03_特征与模型）')
    ap.add_argument('--out', default=None, help='图片输出目录（默认 02_可视化图）')
    args = ap.parse_args()

    global PICK, DATA, MODEL_DIR, FIG_DIR
    PICK = max(0, args.pick)
    if args.data:
        DATA = os.path.abspath(args.data)
    if args.modeldir:
        MODEL_DIR = os.path.abspath(args.modeldir)
    if args.out:
        FIG_DIR = os.path.abspath(args.out)
    want = {s.strip() for s in args.only.split(',')}

    if not os.path.isdir(DATA):
        raise SystemExit('找不到数据目录：%s' % DATA)
    os.makedirs(FIG_DIR, exist_ok=True)
    mapping = cond_wavs()
    print('可用工况：%s' % ('、'.join(sorted(mapping)) or '无'))

    produced = []
    for key, fn in (('1', lambda: fig1(mapping, FIG_DIR)),
                    ('2', lambda: fig2(mapping, FIG_DIR)),
                    ('3', lambda: fig3(mapping, FIG_DIR)),
                    ('4', lambda: fig4(FIG_DIR)),
                    ('5', lambda: fig5(FIG_DIR)),
                    ('6', lambda: fig6(FIG_DIR))):
        if key not in want:
            continue
        p = fn()
        if p:
            produced.append(p)
            print('  已生成 %s' % os.path.basename(p))

    if not produced:
        print('\n一张图都没生成。先采集数据，或运行 python selftest.py 确认管线正常。')
        return 1

    if args.png:
        conv = os.path.join(HERE, 'svg_to_png.py')
        if os.path.isfile(conv):
            subprocess.run([sys.executable, conv] + produced, cwd=HERE)

    print('\n共 %d 张图，输出目录：%s' % (len(produced), FIG_DIR))
    return 0


if __name__ == '__main__':
    sys.exit(main())
