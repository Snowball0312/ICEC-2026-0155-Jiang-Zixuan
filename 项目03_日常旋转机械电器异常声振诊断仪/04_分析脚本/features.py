# -*- coding: utf-8 -*-
"""
features.py —— 批量特征提取：把 01_原始声振数据 里的 WAV / CSV 变成一张特征表

用法
    python features.py                        # 使用默认路径（脚本所在目录的上一级）
    python features.py --root ../01_原始声振数据 --out ../03_特征与模型/features.csv
    python features.py --modality audio       # 只处理麦克风录音
    python features.py --modality vib         # 只处理加速度计 CSV

输出
    03_特征与模型/features.csv
        列：modality, condition, group, gear, distance_cm, file, win_index, t_start, <30 个特征>

命名约定（采集时请严格照此命名，脚本靠它解析标签与分组）
    L2_进风强挡_r07_g1_20cm_1121_1530.wav
    └ 工况码  └ 轮次  └ 档位 └ 距离  └ 日期 └ 时刻
    轮次 r 是交叉验证的分组依据：同一轮的 5 个工况必须同折进同折出，否则会泄漏。
"""

import os
import re
import csv
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DEFAULT = os.path.abspath(os.path.join(HERE, '..', '01_原始声振数据'))
OUT_DEFAULT = os.path.abspath(os.path.join(HERE, '..', '03_特征与模型', 'features.csv'))

WIN_SEC = {'audio': 0.5, 'vib': 1.0}   # 振动采样率低，窗长取 1 s 才够 1 个基频周期以上

# 工况码与轮次之间允许有中文名（可能带下划线），故用 \D* 而非 [^_]*
NAME_PAT = re.compile(r'(L\d)\D*_r(\d+)_g(\d+)_(\d+)cm', re.I)


def parse_name(path):
    """从文件名解析工况码 / 轮次 / 档位 / 距离；解析不到则退回目录名。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = NAME_PAT.search(stem)
    parent = os.path.basename(os.path.dirname(path))
    cond = (m.group(1).upper() if m else parent.split('_')[0])
    group = ('r%02d' % int(m.group(2))) if m else 'r00'
    gear = int(m.group(3)) if m else 0
    dist = int(m.group(4)) if m else 0
    return cond, group, gear, dist


def collect_files(root, modality):
    exts = ('.wav',) if modality == 'audio' else ('.csv',)
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.startswith('采集记录') or fn.lower() in ('features.csv', 'model.json'):
                continue  # 元数据表，不是信号文件
            if fn.lower().endswith(exts):
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description='声振特征批量提取')
    ap.add_argument('--root', default=ROOT_DEFAULT)
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--modality', default='auto', choices=['auto', 'audio', 'vib'])
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print('找不到数据目录：%s' % args.root)
        print('请先把 Phyphox 导出的 WAV / CSV 放进 01_原始声振数据/L0_正常 等子目录。')
        return 2

    modalities = ['audio', 'vib'] if args.modality == 'auto' else [args.modality]
    files = []
    for m in modalities:
        files += [(m, p) for p in collect_files(args.root, m)]
    if not files:
        print('没有找到可处理的文件（%s）' % args.root)
        return 2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = ['modality', 'condition', 'group', 'gear', 'distance_cm',
              'file', 'win_index', 't_start'] + sndlib.FEATURE_NAMES

    rows = 0
    per_file = []
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for modality, path in files:
            try:
                if modality == 'audio':
                    sr, x = sndlib.read_wav(path)
                else:
                    sr, x = sndlib.read_phyphox_csv(path, col=1)
            except Exception as e:
                print('  [跳过] %s -> %s' % (os.path.basename(path), e))
                continue

            cond, group, gear, dist = parse_name(path)
            dur = len(x) / float(sr) if sr else 0.0
            wins = sndlib.extract_windows(x, sr, modality, WIN_SEC[modality], 0.5)
            stems = os.path.splitext(os.path.basename(path))[0]
            for wi, (t0, feats) in enumerate(wins):
                w.writerow([modality, cond, group, gear, dist, stems,
                            wi, '%.3f' % t0] +
                           ['%.6g' % feats[k] for k in sndlib.FEATURE_NAMES])
                rows += 1
            per_file.append((modality, cond, stems, sr, dur, len(wins)))
            print('  %-6s %-3s %-8.0f Hz %6.2f s  %3d 窗  %s' %
                  (modality, cond, sr, dur, len(wins), stems))

    print('\n特征表已写入：%s' % args.out)
    print('文件 %d 个，特征行 %d 行，每行 %d 个特征' %
          (len(per_file), rows, len(sndlib.FEATURE_NAMES)))
    if rows == 0:
        os.remove(args.out)
        print('没有提取到任何特征，已删除空表。请先采集数据或运行 python selftest.py 自检。')
        return 1

    # 汇总每工况文件数，方便核对 5 × 3 是否齐备
    stat = {}
    for modality, cond, _stem, _sr, _dur, _n in per_file:
        stat.setdefault((modality, cond), 0)
        stat[(modality, cond)] += 1
    print('\n%-6s %-10s %s' % ('模态', '工况', '文件数'))
    for (m, c) in sorted(stat):
        print('%-6s %-10s %d' % (m, c, stat[(m, c)]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
