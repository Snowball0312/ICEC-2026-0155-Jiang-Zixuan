# -*- coding: utf-8 -*-
"""
check_dataset.py —— 数据集体检：采集当天收工前跑一遍，缺漏当场补录

用法
    python check_dataset.py
    python check_dataset.py --root ../01_原始声振数据 --expect 6

检查项
    1. 5 个工况目录下的文件数是否达到「每工况 expect 段」
    2. 文件名是否符合 L2_进风强挡_r07_g1_20cm_1121_1530.wav 规范（否则训练时无法解析轮次）
    3. 每条录音时长、采样率、峰值、削波比例、是否近静音
    4. 每个工况目录是否至少有一张实验照（不要求逐段配图）
    5. 每个工况覆盖的轮次数（决定 GroupKFold 的可行性）

输出
    终端汇总表 + 03_特征与模型/数据集体检.md
"""

import os
import re
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DEFAULT = os.path.abspath(os.path.join(HERE, '..', '01_原始声振数据'))
OUT_DEFAULT = os.path.abspath(os.path.join(HERE, '..', '03_特征与模型', '数据集体检.md'))

NAME_PAT = re.compile(r'^(L\d)(_[^_]+)*?(_r\d{2}_g\d_\d+cm)', re.I)
PHOTO_EXT = ('.jpg', '.jpeg', '.png', '.heic')
SKIP = ('features.csv', 'model.json')
# 基准工况每档只录 1 段（一档 + 二档 = 2 段），不按每工况 3 段要求
EXPECT_BY_COND = {'L0': 2}


def scan(root, expect):
    rows = []
    for dirpath, _d, filenames in os.walk(root):
        # 配图按目录判定：目录里出现任意一张实验照，即认为该工况已配图。
        # 一个工况 3 段录音只需 1 张照片，不要求逐段同名。
        dir_photo = any(fn.lower().endswith(PHOTO_EXT) for fn in filenames)
        for fn in sorted(filenames):
            low = fn.lower()
            if not low.endswith(('.wav', '.csv')) or low in SKIP or fn.startswith('采集记录'):
                continue
            path = os.path.join(dirpath, fn)
            stem, ext = os.path.splitext(fn)
            rec = {'file': fn, 'cond': os.path.basename(dirpath), 'ext': ext.lower(),
                   'name_ok': bool(NAME_PAT.match(fn)), 'photo': dir_photo,
                   'sr': 0.0, 'dur': 0.0, 'peak': 0.0, 'clip': 0.0, 'silent': False,
                   'err': ''}
            try:
                if ext.lower() == '.wav':
                    sr, x = sndlib.read_wav(path)
                else:
                    sr, x = sndlib.read_phyphox_csv(path, col=1)
                rec['sr'] = sr
                rec['dur'] = len(x) / float(sr) if sr else 0.0
                peak = max((abs(v) for v in x), default=0.0)
                rec['peak'] = peak
                # WAV 归一化到 ±1，削波判据取 0.99；CSV 为物理量，改用标准差判静音
                if ext.lower() == '.wav':
                    rec['clip'] = sum(1 for v in x if abs(v) > 0.99) / max(1, len(x))
                rms = sndlib._rms(sndlib.remove_dc(x)) if x else 0.0
                rec['silent'] = rms < (1e-3 if ext.lower() == '.wav' else 1e-3)
            except Exception as e:
                rec['err'] = str(e)
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description='数据集体检')
    ap.add_argument('--root', default=ROOT_DEFAULT)
    ap.add_argument('--expect', type=int, default=6)
    ap.add_argument('--out', default=OUT_DEFAULT)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit('找不到数据目录：%s' % args.root)
    rows = scan(args.root, args.expect)
    if not rows:
        raise SystemExit('目录里没有任何 WAV / CSV，先把采集文件放进去。')

    wavs = [r for r in rows if r['ext'] == '.wav']
    csvs = [r for r in rows if r['ext'] == '.csv']
    conds = sorted({r['cond'] for r in rows})

    problems = []
    for r in rows:
        if r['err']:
            problems.append('%s 读取失败：%s' % (r['file'], r['err']))
        if not r['name_ok']:
            problems.append('%s 命名不合规（应为 L1_一档半遮挡_r01_g1_12cm_1122_1416.wav）' % r['file'])
        if r['clip'] > 0.001:
            problems.append('%s 削波 %.1f%%，录近了或音量过大，需重录' % (r['file'], r['clip'] * 100))
        if r['silent']:
            problems.append('%s 近乎静音，检查是否没录上' % r['file'])
        if r['ext'] == '.wav' and r['dur'] < 5:
            problems.append('%s 时长仅 %.1f s，太短' % (r['file'], r['dur']))

    for c in conds:
        if not any(r['photo'] for r in rows if r['cond'] == c):
            problems.append('%s 目录下没有任何实验照（每个工况至少 1 张）' % c)

    lines = []
    lines.append('# 数据集体检报告\n')
    lines.append('检查目录：`01_原始声振数据/`　期望每工况 %d 段\n' % args.expect)
    lines.append('| 工况目录 | WAV 段数 | CSV 段数 | 配图 | 覆盖轮次 | 状态 |')
    lines.append('|---|---|---|---|---|---|')
    for c in conds:
        w = len([r for r in wavs if r['cond'] == c])
        v = len([r for r in csvs if r['cond'] == c])
        ph = len([r for r in rows if r['cond'] == c and r['photo']])
        groups = set()
        for r in rows:
            if r['cond'] == c:
                m = re.search(r'_r(\d{2})_', r['file'])
                if m:
                    groups.add(m.group(1))
        want = EXPECT_BY_COND.get(c.split('_')[0].upper(), args.expect)
        ok = '齐备' if w >= want else '缺 %d 段' % (want - w)
        lines.append('| %s | %d | %d | %d | %d | %s |' % (c, w, v, ph, len(groups), ok))
    lines.append('')
    lines.append('合计：%d 个 WAV、%d 个 CSV、%d 个工况目录。\n' % (len(wavs), len(csvs), len(conds)))

    if problems:
        lines.append('## 待处理问题（%d 条）\n' % len(problems))
        for p in problems[:80]:
            lines.append('- %s' % p)
        if len(problems) > 80:
            lines.append('- …… 另有 %d 条，见终端输出' % (len(problems) - 80))
    else:
        lines.append('## 待处理问题\n\n无。命名、时长、削波、静音、工况配图五项全部通过。\n')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\n报告已写入：%s' % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
