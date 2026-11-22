# -*- coding: utf-8 -*-
"""
predict.py —— 单段录音盲测诊断：读一个 WAV，输出工况结论 + 置信度 + 逐窗明细

用法
    python predict.py ../01_原始声振数据/L2_进风强挡/xxx.wav
    python predict.py xxx.wav --noise-snr 0          # 叠加白噪声到 0 dB 再诊断（抗噪演示）
    python predict.py xxx.wav --detail ../03_特征与模型/单段逐窗明细.csv
    python predict.py xxx.wav --model ../03_特征与模型/model.json

输出
    终端打印：结论工况、置信度、Top-3 备选、每类投票占比、是否可信
    --detail 指定时另存逐窗概率表，便于在报告里画证据链

盲测现场就用这一条命令：老师随机设工况 → 你录 30 s → 直接跑 → 屏幕出结论。
"""

import os
import sys
import csv
import json
import math
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT = os.path.abspath(os.path.join(HERE, '..', '03_特征与模型', 'model.json'))


def add_white_noise(x, snr_db, seed=0):
    """按目标信噪比叠加白噪声（按信号 RMS 定标）。"""
    if snr_db is None:
        return x
    ps = sum(v * v for v in x) / max(1, len(x))
    if ps <= 1e-20:
        return x
    pn = ps / (10 ** (snr_db / 10.0))
    sigma = math.sqrt(pn)
    rng = random.Random(seed)
    return [v + rng.gauss(0.0, sigma) for v in x]


def posterior(model, x):
    z = [(v - model['mean'][j]) / (model['std'][j] or 1.0) for j, v in enumerate(x)]
    W, b = model['W'], model['b']
    logits = [sum(W[j][c] * z[j] for j in range(len(z))) + b[c]
              for c in range(len(model['classes']))]
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v / s for v in e]


def diagnose(model, path, noise_snr=None, win_sec=None, verbose=True):
    sr, x = sndlib.read_wav(path)
    if noise_snr is not None:
        x = add_white_noise(x, noise_snr)
    ws = win_sec or (0.5 if model['modality'] == 'audio' else 1.0)
    wins = sndlib.extract_windows(x, sr, model['modality'], ws, 0.5)

    classes = model['classes']
    rows = []
    agg = [0.0] * len(classes)
    for t0, feats in wins:
        vec = [feats[k] for k in model['features']]
        p = posterior(model, vec)
        rows.append((t0, p, classes[max(range(len(p)), key=lambda i: p[i])]))
        for c in range(len(classes)):
            agg[c] += p[c]
    if not rows:
        raise SystemExit('录音太短，切不出一个分析窗：%s' % path)

    share = [v / len(rows) for v in agg]
    best = max(range(len(classes)), key=lambda c: share[c])
    # 逐窗硬投票占比作为稳定性指标
    hard = [0] * len(classes)
    for _t, _p, lab in rows:
        hard[classes.index(lab)] += 1
    stability = hard[best] / len(rows)
    order = sorted(range(len(classes)), key=lambda c: -share[c])

    if verbose:
        print('\n文件　　　：%s' % os.path.basename(path))
        print('采样率　　：%.0f Hz　时长：%.2f s　分析窗：%d 个 × %.2f s' %
              (sr, len(x) / float(sr), len(rows), ws))
        if noise_snr is not None:
            print('抗噪设置　：叠加白噪声至 %g dB SNR' % noise_snr)
        print('─' * 52)
        print('诊断结论　：%s' % classes[best])
        print('置信度　　：%.1f%%　（逐窗硬投票一致率 %.1f%%）' %
              (share[best] * 100, stability * 100))
        print('各工况占比：')
        for c in order:
            bar = '█' * int(round(share[c] * 24))
            print('  %-10s %5.1f%% %s' % (classes[c], share[c] * 100, bar))
        if len(order) > 1 and share[order[1]] > 0.25:
            print('⚠ 次高工况 %s 占比 %.1f%%，结论存疑，建议延长录音或换位置复测。'
                  % (classes[order[1]], share[order[1]] * 100))
        print('─' * 52)
    return classes[best], share[best], share, rows


def main():
    ap = argparse.ArgumentParser(description='单段录音盲测诊断')
    ap.add_argument('wav')
    ap.add_argument('--model', default=MODEL_DEFAULT)
    ap.add_argument('--noise-snr', type=float, default=None)
    ap.add_argument('--detail', default=None)
    ap.add_argument('--win', type=float, default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit('找不到模型：%s\n请先运行 python features.py && python train.py' % args.model)
    with open(args.model, 'r', encoding='utf-8') as f:
        model = json.load(f)

    label, conf, share, rows = diagnose(model, args.wav, args.noise_snr, args.win)

    if args.detail:
        os.makedirs(os.path.dirname(os.path.abspath(args.detail)), exist_ok=True)
        with open(args.detail, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['窗序号', '起始时间_s'] + ['P_' + c for c in model['classes']] + ['判决'])
            for i, (t0, p, lab) in enumerate(rows):
                w.writerow([i, '%.3f' % t0] + ['%.4f' % v for v in p] + [lab])
        print('逐窗明细已写入：%s' % args.detail)
    return 0


if __name__ == '__main__':
    sys.exit(main())
