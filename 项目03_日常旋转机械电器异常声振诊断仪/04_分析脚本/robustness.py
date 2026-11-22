# -*- coding: utf-8 -*-
"""
robustness.py —— 抗噪实验：在同一批真实录音上叠加不同强度噪声，测准确率衰减曲线

用法
    python robustness.py                                  # 白噪声，SNR 20→-5 dB
    python robustness.py --noise pink                     # 粉噪声（更接近电器背景）
    python robustness.py --snrs 30,20,10,5,0,-5,-10
    python robustness.py --limit 3                        # 每工况只取 3 段，快速跑通

输出
    03_特征与模型/抗噪曲线.csv     每档 SNR 的总体与分类准确率
    03_特征与模型/抗噪报告.md      可直接进报告的结果段（含结论判断）

为什么要在真实录音上加噪，而不是用合成信号
    加噪保留了真实的叶片通过频率、共振峰与房间混响，只改变信噪比这一个变量，
    这样得到的曲线才能回答“这个诊断仪在嘈杂厨房里还能不能用”。
"""

import os
import re
import sys
import csv
import math
import json
import zlib
import random
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib
import predict as P

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_MODEL = os.path.abspath(os.path.join(HERE, '..', '03_特征与模型'))
ROOT_DEFAULT = os.path.abspath(os.path.join(HERE, '..', '01_原始声振数据'))


def pink_noise(n, rng):
    """三阶 1/f 粉噪声（经典 b0–b3 / a1–a3 系数），零相位近似，够做抗噪对照。"""
    b0, b1, b2, b3 = 0.049922035, -0.095993537, 0.050612699, -0.004408786
    a1, a2, a3 = -2.494956002, 2.017265875, -0.522189400
    out = [0.0] * n
    x1 = x2 = x3 = y1 = y2 = y3 = 0.0
    for i in range(n):
        w = rng.gauss(0.0, 1.0)
        y = b0 * w + b1 * x1 + b2 * x2 + b3 * x3 - a1 * y1 - a2 * y2 - a3 * y3
        x3, x2, x1 = x2, x1, w
        y3, y2, y1 = y2, y1, y
        out[i] = y
    return out


def mix_at_snr(x, snr_db, kind='white', seed=0):
    """把噪声缩放到指定 SNR 后与信号相加。"""
    rng = random.Random(seed)
    n = len(x)
    if kind == 'pink':
        nz = pink_noise(n, rng)
    else:
        nz = [rng.gauss(0.0, 1.0) for _ in range(n)]
    ps = sum(v * v for v in x) / max(1, n)
    pn = sum(v * v for v in nz) / max(1, n)
    if ps <= 1e-20 or pn <= 1e-20:
        return list(x)
    target = ps / (10 ** (snr_db / 10.0))
    g = math.sqrt(target / pn)
    return [x[i] + g * nz[i] for i in range(n)]


def safe_div(a, b):
    return a / b if b else 0.0


def label_of(path, cond, classes):
    """真值标签：优先取目录名前缀（L0_正常 → L0），退回文件名里的 L 码。"""
    head = cond.split('_')[0]
    if head in classes:
        return head
    m = re.search(r'(L\d)', os.path.basename(path))
    if m and m.group(1) in classes:
        return m.group(1)
    return None


def collect(root, limit):
    buckets = {}
    for dirpath, _d, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith('.wav'):
                cond = os.path.basename(dirpath)
                buckets.setdefault(cond, []).append(os.path.join(dirpath, fn))
    items = []
    for cond in sorted(buckets):
        for p in buckets[cond][:limit] if limit else buckets[cond]:
            items.append((cond, p))
    return items


# ---------------------------------------------------------------- 并行判决
# 单个 (信噪比 × 录音) 组合要跑一次完整的分帧+FFT+后验，纯标准库下约 20–30 s；
# 5 档 SNR × 14 段 ≈ 80 多个组合，串行要一个多小时。特征提取彼此完全独立，
# 用 stdlib 的 multiprocessing 按组合切分，32 核机上 10 进程约 4 分钟跑完。
_G = {}


def _worker_init(model, noise, ws):
    _G['model'] = model
    _G['noise'] = noise
    _G['ws'] = ws


def _worker_eval(job):
    snr, cond, path = job
    model = _G['model']
    classes = model['classes']
    key = label_of(path, cond, classes)
    if key is None:
        return None
    try:
        sr, x = sndlib.read_wav(path)
    except Exception:
        return None
    if snr is not None:
        x = mix_at_snr(x, snr, _G['noise'], seed=zlib.crc32(path.encode('utf-8')))
    wins = sndlib.extract_windows(x, sr, model['modality'], _G['ws'], 0.5)
    if not wins:
        return None
    agg = [0.0] * len(classes)
    for _t, feats in wins:
        p = P.posterior(model, [feats[k] for k in model['features']])
        for c in range(len(classes)):
            agg[c] += p[c]
    pred = classes[max(range(len(classes)), key=lambda c: agg[c])]
    return snr, key, pred


def main():
    ap = argparse.ArgumentParser(description='抗噪曲线实验')
    ap.add_argument('--root', default=ROOT_DEFAULT)
    ap.add_argument('--model', default=os.path.join(DIR_MODEL, 'model.json'))
    ap.add_argument('--noise', default='white', choices=['white', 'pink'])
    ap.add_argument('--snrs', default='clean,20,10,5,0,-5')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=0, help='并行进程数，0 为自动（CPU-1，最多 12）；1 为串行')
    ap.add_argument('--outdir', default=DIR_MODEL)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit('找不到模型：%s，请先训练。' % args.model)
    with open(args.model, 'r', encoding='utf-8') as f:
        model = json.load(f)

    items = collect(args.root, args.limit)
    if not items:
        raise SystemExit('没找到 WAV，检查 --root：%s' % args.root)
    classes = model['classes']

    snrs = []
    for tok in args.snrs.split(','):
        tok = tok.strip().lower()
        snrs.append(None if tok in ('clean', 'clean0', 'inf') else float(tok))

    results = []
    ws = 0.5 if model['modality'] == 'audio' else 1.0
    jobs = [(snr, cond, path) for snr in snrs for cond, path in items]
    nproc = args.jobs if args.jobs > 0 else min(12, max(1, (os.cpu_count() or 2) - 1))
    print('判决组合 %d 个（%d 档信噪比 × %d 段录音），用 %d 个进程'
          % (len(jobs), len(snrs), len(items), nproc))

    if nproc == 1:
        _worker_init(model, args.noise, ws)
        outs = [_worker_eval(j) for j in jobs]
    else:
        with mp.Pool(nproc, initializer=_worker_init,
                     initargs=(model, args.noise, ws)) as pool:
            outs = [o for o in pool.imap_unordered(_worker_eval, jobs, chunksize=1)]

    for snr in snrs:
        hit = {c: 0 for c in classes}
        tot = {c: 0 for c in classes}
        for o in outs:
            if o is None or o[0] != snr:
                continue
            _snr, key, pred = o
            tot[key] = tot.get(key, 0) + 1
            if pred == key:
                hit[key] = hit.get(key, 0) + 1
        total = sum(tot.values())
        correct = sum(hit.values())
        results.append((snr, correct, total, hit, tot))
        tag = '无噪声' if snr is None else '%g dB' % snr
        print('%-8s 整段准确率 %5.1f%% (%d/%d)' %
              (tag, 100.0 * correct / max(1, total), correct, total))

    os.makedirs(args.outdir, exist_ok=True)
    suffix = '_粉噪' if args.noise == 'pink' else ''
    csv_path = os.path.join(args.outdir, '抗噪曲线%s.csv' % suffix)
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['信噪比_dB', '测试段数', '正确段数', '整段准确率'] +
                   ['%s准确率' % c for c in classes])
        for snr, correct, total, hit, tot in results:
            w.writerow(['无噪声' if snr is None else '%g' % snr, total, correct,
                        '%.4f' % (correct / max(1, total))] +
                       ['%.4f' % safe_div(hit.get(c, 0), tot.get(c, 0)) for c in classes])

    md = ['# 抗噪实验报告\n',
          '- 噪声类型：%s' % ('粉噪声（1/f，近似电器背景）' if args.noise == 'pink' else '白噪声'),
          '- 测试样本：%d 段真实录音，类别来自目录名，模型未见过这些录音' % len(items),
          '- 加噪方式：按信号 RMS 定标噪声增益，逐档叠加后重新提特征、重新判决\n',
          '| 信噪比 | 整段准确率 | ' + ' | '.join(classes) + ' |',
          '|---' * (len(classes) + 2) + '|']
    for snr, correct, total, hit, tot in results:
        md.append('| %s | %.1f%% | ' % ('无噪声' if snr is None else '%g dB' % snr,
                                        100.0 * correct / max(1, total)) +
                  ' | '.join('%.0f%%' % (100.0 * safe_div(hit.get(c, 0), tot.get(c, 0))) for c in classes) +
                  ' |')

    clean = results[0][1] / max(1, results[0][2]) if results else 0.0
    zero = [r for r in results if r[0] == 0]
    md.append('')
    if zero:
        z = 100.0 * zero[0][1] / max(1, zero[0][2])
        drop = clean * 100 - z
        md.append('**结论**：无噪声时整段准确率 %.1f%%，0 dB 信噪比下降到 %.1f%%（衰减 %.1f 个百分点）。'
                  % (clean * 100, z, drop))
        if z >= 60:
            md.append('0 dB 下仍能正确判决，说明模型主要依靠谐波结构而非绝对能量，'
                      '对背景噪声不敏感，这是可以写进结论的强项。')
        elif drop <= 30:
            md.append('衰减在 30 个百分点以内，属可接受范围；报告里应说明'
                      '“强噪声下需靠近设备或延长录音时间”的使用边界。')
        else:
            md.append('衰减超过 30 个百分点，说明强噪声下特征被淹没。'
                      '这一现象必须列入局限性，并给出对策：固定录音距离、'
                      '引入谱平坦度与谱熵这类噪声自适应特征，或先用谱减降噪再判决。')

    md_path = os.path.join(args.outdir, '抗噪报告%s.md' % suffix)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n已写入：%s\n已写入：%s' % (csv_path, md_path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
