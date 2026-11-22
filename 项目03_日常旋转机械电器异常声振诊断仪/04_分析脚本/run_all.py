# -*- coding: utf-8 -*-
"""
run_all.py —— 一条命令跑完整条流水线：体检 → 特征 → 训练 → 抗噪 → 出图 → 汇总

用法
    python run_all.py                 # 全流程（数据采集完成后用这个）
    python run_all.py --modality vib  # 只做加速度模态
    python run_all.py --skip 抗噪      # 跳过某一步（抗噪较慢，可先跳过）
    python run_all.py --limit 3       # 抗噪每工况只取 3 段，快速过一遍

各步独立子进程执行，某步失败会停下并打印该步输出，不会掩盖错误。
最后写一份 03_特征与模型/一键结果.md，把各步关键数字汇总到一处，方便直接誊进报告。
"""

import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
MODEL_DIR = os.path.join(PROJ, '03_特征与模型')

STEPS = [
    ('体检', ['check_dataset.py', '--expect', '3'], '数据集体检：段数、命名、配图、削波'),
    ('特征', ['features.py'], '提取 32 维特征，生成 features.csv'),
    ('训练', ['train.py', '--baseline', 'L0'], '收缩 LDA + 按轮次 GroupKFold，L0 仅作基准'),
    ('抗噪', ['robustness.py', '--noise', 'white',
              '--snrs', 'clean,20,15,12,10,5,0,-5'],
     '白噪声抗噪曲线 8 档（并行，约 4 分钟）'),
    ('出图', ['plot_figs.py', '--png'], '6 张图（SVG + PNG），输出到 02_可视化图'),
]


def run_step(script, args):
    cmd = [sys.executable, script] + args
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def grab(text, keys):
    out = []
    for line in text.splitlines():
        if line.strip().startswith('#'):
            continue
        for k in keys:
            if k in line:
                out.append(line.strip())
                break
    return out


def main():
    ap = argparse.ArgumentParser(description='一键跑完整流水线')
    ap.add_argument('--modality', default='audio', choices=['audio', 'vib'])
    ap.add_argument('--skip', default='', help='跳过的步骤名，逗号分隔，如 抗噪,体检')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--data', default=None, help='原始数据目录（默认 01_原始声振数据）')
    ap.add_argument('--modeldir', default=None, help='结果目录（默认 03_特征与模型）')
    args = ap.parse_args()

    global MODEL_DIR
    if args.modeldir:
        MODEL_DIR = os.path.abspath(args.modeldir)
    data = os.path.abspath(args.data) if args.data else os.path.join(PROJ, '01_原始声振数据')
    skip = {s.strip() for s in args.skip.split(',') if s.strip()}
    results = []

    for name, base, desc in STEPS:
        if name in skip:
            print('── %s：已跳过' % name)
            continue
        cmd_args = base[1:]
        if name in ('体检', '特征'):
            cmd_args += ['--root', data] if name == '体检' else ['--root', data,
                                                                 '--out', os.path.join(MODEL_DIR, 'features.csv')]
            if name == '体检':
                cmd_args += ['--out', os.path.join(MODEL_DIR, '数据集体检.md')]
        if name == '训练':
            cmd_args += ['--modality', args.modality,
                         '--features', os.path.join(MODEL_DIR, 'features.csv'),
                         '--outdir', MODEL_DIR]
        if name == '抗噪':
            cmd_args += ['--model', os.path.join(MODEL_DIR, 'model.json'),
                         '--root', data, '--outdir', MODEL_DIR]
            if args.limit:
                cmd_args += ['--limit', str(args.limit)]
        if name == '出图':
            cmd_args = ['--png', '--data', data, '--modeldir', MODEL_DIR,
                        '--out', os.path.join(PROJ, '02_可视化图')]

        print('── %s：%s' % (name, desc))
        code, out = run_step(base[0], cmd_args)
        tail = [l for l in out.strip().splitlines() if l.strip()][-4:]
        for l in tail:
            print('     %s' % l)
        results.append((name, code, out))
        if code != 0:
            print('\n[中止] %s 失败，退出码 %d。完整输出：\n' % (name, code))
            print(out)
            return code

    # ---- 汇总
    lines = ['# 一键流水线结果\n', '运行环境：Python %s\n' % sys.version.split()[0]]
    facts = {
        '体检': ['合计：', '待处理问题', '无。命名'],
        '特征': ['特征行', '文件 '],
        '训练': ['样本：', '窗口级准确率', '折 ', '同档位遮挡判别', '  '],
        '抗噪': ['整段准确率'],
        '出图': ['共 ', '已生成'],
    }
    for name, code, out in results:
        got = grab(out, facts.get(name, []))
        lines.append('## %s\n' % name)
        if got:
            for g in got[:14]:
                lines.append('- %s' % g)
        else:
            lines.append('- （无自动摘录，见终端输出）')
        lines.append('')

    path = os.path.join(MODEL_DIR, '一键结果.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n全部完成，汇总已写入：%s' % path)
    print('下一步：把 02_可视化图 的图与 03_特征与模型 的结果誊进 06_诊断报告。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
