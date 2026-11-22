# -*- coding: utf-8 -*-
"""gear_check.py —— 档位基线核查（一次连录多档，自动切段并读主音）

为什么需要这个脚本
------------------
建档时"一档/二档"是靠"按了几次键"记的。事后发现这台风扇实为三档、
相邻两档只差约 10%，而遮挡会把低档转速拽低三分之一——于是"标称档位"与
"实测转速"经常对不上（详见诊断报告 5.3 节）。

本脚本做三件事，全部可复算：
  1. 用一阶差分能量找按键的"咔哒"瞬态（宽带、短促），据此把一次连录切成 N 段；
  2. 每段取 400–2000 Hz 内最强峰，作为"主音"（与建档时读 BPF 的口径一致）；
  3. 每段另打印最强 8 个谱峰，用来人工核对谐波族（基频 = 相邻谐波的公共间隔）。

口径说明：主音是"最响的那个音"，会随麦克风位置在 6～24 阶谐波之间漂移；
跨录音比较转速时应看"谐波族基频"（各阶的公共间隔），不要直接比主音。
脚本不自动推断基频——谐波族缺阶时自动推断会给出误导性结果，这里只给证据。

用法
----
    python gear_check.py <wav> [更多 wav ...]
    python gear_check.py            # 默认核查 现场录音/档位验证_20261122.wav
"""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib as S

CLICK_K = 6.0      # 瞬态判据：差分能量 > 中位数的 6 倍
CLICK_GAP = 0.25   # 两次按键的最小间隔（s）
GUARD = 0.12       # 切段后跳过按键本身的余振（s）
TONE_LO, TONE_HI = 400.0, 2000.0
PEAK_TOP = 8


def find_clicks(x, sr):
    """返回按键瞬态的时刻列表（秒）。"""
    d = [x[i] - x[i - 1] for i in range(1, len(x))]
    n = max(1, int(0.01 * sr))
    rows = [(i / sr, S._rms(d[i:i + n])) for i in range(0, len(d) - n, n)]
    med = statistics.median([r[1] for r in rows]) or 1e-9
    out = []
    for t, r in rows:
        if r > med * CLICK_K and (not out or t - out[-1] > CLICK_GAP):
            out.append(t)
    return out


def top_peaks(freqs, mags, fmax=3600.0, min_gap=25.0, limit=PEAK_TOP):
    cand = []
    for k in range(2, len(mags) - 2):
        f = freqs[k]
        if f < 55.0 or f > fmax:
            continue
        if mags[k] > mags[k - 1] and mags[k] >= mags[k + 1]:
            cand.append((mags[k], f))
    cand.sort(reverse=True)
    out = []
    for m, f in cand:
        if all(abs(f - g) > min_gap for g, _ in out):
            out.append((f, m))
        if len(out) >= limit:
            break
    return out


def analyse(path):
    sr, x = S.read_wav(path)
    dur = len(x) / sr
    clicks = find_clicks(x, sr)
    print('文件：%s' % os.path.basename(path))
    print('时长 %.2f s，采样率 %d Hz，检测到 %d 次按键：%s' %
          (dur, sr, len(clicks), ' / '.join('%.2f s' % c for c in clicks)))
    print()

    bounds = [0.0] + clicks + [dur]
    rows = []
    for i in range(len(bounds) - 1):
        t0 = bounds[i] + GUARD
        t1 = bounds[i + 1]
        if clicks and t0 < clicks[0]:
            continue  # 第一次按键之前是环境底噪，不算工况
        if t1 - t0 < 0.8:
            continue
        a, b = int(t0 * sr), int(t1 * sr)
        seg = x[a:b]
        freqs, mags = S.spectrum(seg, sr)
        tone, _ = S.dominant_freq(freqs, mags, TONE_LO, TONE_HI)
        pk = top_peaks(freqs, mags)
        rms = S._rms(seg)
        rows.append({'i': i, 't0': t0, 't1': t1, 'tone': tone, 'rms': rms, 'pk': pk})
        print('段 %d  %.2f – %.2f s（%.2f s）  主音 %6.1f Hz  段RMS %.4f（%.1f dBFS）' %
              (i, t0, t1, t1 - t0, tone, rms, S.db(rms)))
        print('      最强峰：' + '  '.join('%.0f:%.0f' % (f, m) for f, m in pk))
    print()
    print('提示：主音两侧若出现等间隔的峰（如 660/1320/1980），该间隔即谐波族基频；')
    print('      跨录音比较转速请用这个间隔，不要用主音本身。')
    return rows


def main(argv):
    if not argv:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        argv = [os.path.join(here, '05_盲测与抗噪', '现场录音', '档位验证_20261122.wav')]
    for p in argv:
        if not os.path.exists(p):
            print('找不到文件：%s' % p)
            continue
        analyse(p)
        print('-' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
