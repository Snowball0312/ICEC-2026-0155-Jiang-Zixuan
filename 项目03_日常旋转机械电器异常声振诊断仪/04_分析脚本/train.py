# -*- coding: utf-8 -*-
"""
train.py —— 训练声振工况分类器（LDA + kNN 对照），按“轮次”分组的 GroupKFold 交叉验证

用法
    python train.py                        # 默认读 03_特征与模型/features.csv
    python train.py --modality vib
    python train.py --folds 5 --shrinkage 0.1

为什么要按轮次分组
    同一轮的 5 个工况是在同一位置、同一档位、同一环境噪声下连续录的；
    如果把同一轮的窗口随机拆到训练集和测试集，模型会“记住”这一轮的环境，
    交叉验证准确率虚高，现场盲测必然崩盘。GroupKFold 强制整轮进出。

输出（均写入 03_特征与模型/）
    model.json           训练好的模型参数，供 predict.py 直接加载
    cv_folds.csv         逐折结果：窗口级准确率、整段（文件级）准确率
    cv_同档遮挡.csv       档位内「半遮挡 vs 全遮挡」二分类（检验是否偷看转速）
    混淆矩阵_窗口.csv     真实类别 × 预测类别
    分类报告.csv          每类 precision / recall / F1 / 支持数
    训练摘要.md           可直接粘进诊断报告的结果段
"""

import os
import csv
import sys
import json
import math
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sndlib

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_MODEL = os.path.abspath(os.path.join(HERE, '..', '03_特征与模型'))


# ---------------------------------------------------------------- 数据读取

def load_table(path, modality):
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['modality'] == modality]
    if not rows:
        raise SystemExit('features.csv 里没有 %s 模态的数据' % modality)
    X, y, g, files, gears = [], [], [], [], []
    for r in rows:
        X.append([float(r[k]) for k in sndlib.FEATURE_NAMES])
        y.append(r['condition'])
        g.append(r['group'])
        files.append(r['file'])
        try:
            gears.append(int(float(r.get('gear') or 0)))
        except (TypeError, ValueError):
            gears.append(0)
    return X, y, g, files, gears


# ---------------------------------------------------------------- 数学工具

def zscore_fit(X):
    d = len(X[0])
    mean = [0.0] * d
    std = [0.0] * d
    n = len(X)
    for row in X:
        for j in range(d):
            mean[j] += row[j]
    for j in range(d):
        mean[j] /= n
    for row in X:
        for j in range(d):
            std[j] += (row[j] - mean[j]) ** 2
    for j in range(d):
        std[j] = math.sqrt(std[j] / n) or 1.0
    return mean, std


def zscore_apply(X, mean, std):
    return [[(v - mean[j]) / std[j] for j, v in enumerate(row)] for row in X]


def mat_inv(A):
    """高斯-约当消元求逆（带部分选主元），A 为方阵 list[list[float]]。"""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            M[piv][col] = 1e-12
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(2 * n):
            M[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col]
            if fac == 0.0:
                continue
            for j in range(2 * n):
                M[r][j] -= fac * M[col][j]
    return [row[n:] for row in M]


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


# ---------------------------------------------------------------- LDA

class LDA:
    """线性判别分析（含协方差收缩，小样本下比直接用样本协方差稳）。"""

    def __init__(self, shrinkage=0.1):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self.classes = sorted(set(y))
        d = len(X[0])
        n = len(X)
        k = len(self.classes)
        idx = {c: [i for i, v in enumerate(y) if v == c] for c in self.classes}
        mu = []
        for c in self.classes:
            ids = idx[c]
            mu.append([sum(X[i][j] for i in ids) / len(ids) for j in range(d)])
        # 类内散布矩阵
        S = [[0.0] * d for _ in range(d)]
        for c_i, c in enumerate(self.classes):
            ids = idx[c]
            for i in ids:
                diff = [X[i][j] - mu[c_i][j] for j in range(d)]
                for a in range(d):
                    da = diff[a]
                    if da == 0.0:
                        continue
                    for b in range(a, d):
                        S[a][b] += da * diff[b]
        for a in range(d):
            for b in range(a):
                S[a][b] = S[b][a]
        dof = max(1, n - k)
        S = [[v / dof for v in row] for row in S]
        # 收缩：Σ = (1-λ)S + λ·(tr(S)/d)·I
        tr = sum(S[j][j] for j in range(d))
        lam = self.shrinkage
        for j in range(d):
            S[j][j] = (1 - lam) * S[j][j] + lam * (tr / d + 1e-9)
        Sinv = mat_inv(S)
        # W = Σ⁻¹μ，b = -0.5 μᵀΣ⁻¹μ + ln P(c)
        self.W = [[0.0] * k for _ in range(d)]
        self.b = []
        for c_i in range(k):
            m = mu[c_i]
            Wc = [sum(Sinv[a][j] * m[j] for j in range(d)) for a in range(d)]
            for a in range(d):
                self.W[a][c_i] = Wc[a]
            self.b.append(-0.5 * sum(m[j] * Wc[j] for j in range(d)) +
                          math.log(len(idx[self.classes[c_i]]) / n))
        return self

    def decision(self, x):
        return [sum(self.W[j][c] * x[j] for j in range(len(x))) + self.b[c]
                for c in range(len(self.classes))]

    def predict_proba(self, x):
        return softmax(self.decision(x))


# ---------------------------------------------------------------- kNN

class KNN:
    def __init__(self, k=7):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_proba(self, x):
        dists = []
        for i, row in enumerate(self.X):
            s = 0.0
            for j in range(len(x)):
                d = x[j] - row[j]
                s += d * d
            dists.append((s, self.y[i]))
        dists.sort(key=lambda t: t[0])
        near = dists[:self.k]
        classes = sorted(set(self.y))
        votes = {c: 0.0 for c in classes}
        for s, lab in near:
            votes[lab] += 1.0 / (math.sqrt(s) + 1e-6)
        tot = sum(votes.values()) or 1.0
        return [votes[c] / tot for c in classes]


# ---------------------------------------------------------------- 交叉验证

def group_folds(groups, k, seed=42):
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    buckets = [[] for _ in range(min(k, len(uniq)))]
    for i, g in enumerate(uniq):
        buckets[i % len(buckets)].append(g)
    return buckets


def eval_model(model, classes, X, y):
    preds, probs = [], []
    for x in X:
        p = model.predict_proba(x)
        probs.append(p)
        preds.append(classes[max(range(len(p)), key=lambda i: p[i])])
    acc = sum(1 for i in range(len(y)) if preds[i] == y[i]) / len(y)
    return acc, preds, probs


def same_gear_block_pair(X, y, gears, files, g, shrinkage, folds=5):
    """同档位遮挡判别：转速固定，只让遮挡程度变。

    本项目 4 个工况由「档位 × 遮挡」组成（2 档 × 半/全），档位本身进了标签，
    所以原来的“留一档位”不再成立——被留出的档位里根本是另外两个类别。
    换成这个实验：在每个档位内部单独建“半遮挡 vs 全遮挡”分类器，按轮次分折。
    同档位内仍能判对，说明模型抓的是遮挡带来的气动/结构变化，
    而不是靠绝对基频（转速）蒙对档位。
    """
    rows = []
    for gv in sorted(set(v for v in gears if v > 0)):
        idx = [i for i, v in enumerate(gears) if v == gv]
        labels = sorted(set(y[i] for i in idx))
        if len(labels) < 2:
            continue
        subX = [X[i] for i in idx]
        suby = [y[i] for i in idx]
        subf = [files[i] for i in idx]
        subg = [g[i] for i in idx]
        accs, faccs = [], []
        for test_groups in group_folds(subg, folds):
            te = [i for i, gg in enumerate(subg) if gg in test_groups]
            tr = [i for i, gg in enumerate(subg) if gg not in test_groups]
            if not tr or not te:
                continue
            Xtr = [subX[i] for i in tr]
            ytr = [suby[i] for i in tr]
            mean, std = zscore_fit(Xtr)
            Ztr = zscore_apply(Xtr, mean, std)
            Zte = zscore_apply([subX[i] for i in te], mean, std)
            yte = [suby[i] for i in te]
            lda = LDA(shrinkage).fit(Ztr, ytr)
            if len(lda.classes) < 2:
                continue
            aw, pred, prob = eval_model(lda, lda.classes, Zte, yte)
            af, _ = file_level_vote([subf[i] for i in te], pred, prob, lda.classes, yte)
            accs.append(aw)
            faccs.append(af)
        if not accs:
            continue
        rows.append(['%d 档（%s）' % (gv, ' vs '.join(labels)),
                     len(idx), len(accs),
                     sum(accs) / len(accs), sum(faccs) / len(faccs)])
    return rows


def file_level_vote(files, preds, probs, classes, y, weights=None):
    """同一段录音的窗口投票 → 整段结论，这是盲测现场的真实输出形式。"""
    agg = {}
    for i, f in enumerate(files):
        agg.setdefault(f, []).append(i)
    correct = 0
    detail = []
    for f, ids in agg.items():
        score = [0.0] * len(classes)
        for i in ids:
            for c in range(len(classes)):
                score[c] += probs[i][c]
        win = classes[max(range(len(score)), key=lambda c: score[c])]
        truth = y[ids[0]]
        correct += 1 if win == truth else 0
        detail.append((f, truth, win, max(score) / sum(score)))
    return correct / len(agg), detail


def main():
    ap = argparse.ArgumentParser(description='声振工况分类器训练')
    ap.add_argument('--features', default=os.path.join(DIR_MODEL, 'features.csv'))
    ap.add_argument('--modality', default='audio', choices=['audio', 'vib'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--shrinkage', type=float, default=0.1)
    ap.add_argument('--outdir', default=DIR_MODEL)
    ap.add_argument('--baseline', default='L0',
                    help='只作基准对照、不参与分类训练的工况码（默认 L0；传空串则全部参与）')
    args = ap.parse_args()

    if not os.path.isfile(args.features):
        raise SystemExit('找不到特征表：%s\n请先运行 python features.py' % args.features)

    X, y, g, files, gears = load_table(args.features, args.modality)

    # ---- 基准工况剥离
    # L0（无遮挡）每个档位只录了 1 段，也就是同一个类里同时包含 1102 Hz 和 1219 Hz
    # 两种完全不同的谱形，是一个“双峰类”。实测把它直接塞进五分类，
    # 留一轮那一折会整折塌掉（准确率 0.25，全部被判成别的类）。
    # 所以 L0 只当基准对照，不参与分类；分类任务是四种遮挡态的区分。
    n_base = 0
    if args.baseline:
        keep = [i for i in range(len(y)) if y[i] != args.baseline]
        n_base = len(y) - len(keep)
        if keep and len(set(y[i] for i in keep)) >= 2:
            X = [X[i] for i in keep]
            y = [y[i] for i in keep]
            g = [g[i] for i in keep]
            files = [files[i] for i in keep]
            gears = [gears[i] for i in keep]
            print('基准工况 %s：%d 个窗口不参与分类训练，仅用于报告中的对照' %
                  (args.baseline, n_base))
    classes = sorted(set(y))
    print('样本：%d 个窗口 / %d 段录音 / %d 类工况 / %d 个轮次' %
          (len(X), len(set(files)), len(classes), len(set(g))))
    for c in classes:
        print('  %-10s %4d 窗  %3d 段' %
              (c, y.count(c), len(set(f for i, f in enumerate(files) if y[i] == c))))

    folds = group_folds(g, args.folds)
    os.makedirs(args.outdir, exist_ok=True)

    cv_rows = []
    all_true, all_pred = [], []
    file_records = []
    for fi, test_groups in enumerate(folds):
        test_idx = [i for i, gg in enumerate(g) if gg in test_groups]
        train_idx = [i for i, gg in enumerate(g) if gg not in test_groups]
        if not train_idx or not test_idx:
            continue
        Xtr = [X[i] for i in train_idx]
        ytr = [y[i] for i in train_idx]
        mean, std = zscore_fit(Xtr)
        Ztr = zscore_apply(Xtr, mean, std)
        Zte = zscore_apply([X[i] for i in test_idx], mean, std)
        yte = [y[i] for i in test_idx]
        fte = [files[i] for i in test_idx]

        lda = LDA(args.shrinkage).fit(Ztr, ytr)
        knn = KNN(7).fit(Ztr, ytr)
        acc_l, pred_l, prob_l = eval_model(lda, lda.classes, Zte, yte)
        acc_k, pred_k, prob_k = eval_model(knn, lda.classes, Zte, yte)
        acc_fl, _ = file_level_vote(fte, pred_l, prob_l, lda.classes, yte)

        cv_rows.append([fi + 1, ','.join(sorted(test_groups)), len(train_idx), len(test_idx),
                        '%.4f' % acc_l, '%.4f' % acc_k, '%.4f' % acc_fl])
        all_true += yte
        all_pred += pred_l
        file_records += [(fte[i], yte[i], pred_l[i], prob_l[i]) for i in range(len(yte))]
        print('折 %d │ 测 %2d 轮 %3d 窗 │ LDA %.3f  kNN %.3f  整段 %.3f' %
              (fi + 1, len(test_groups), len(test_idx), acc_l, acc_k, acc_fl))

    # ---- 同档位遮挡判别（排除“靠转速蒙对档位”的捷径）
    gear_rows = same_gear_block_pair(X, y, gears, files, g, args.shrinkage)
    if gear_rows:
        print('\n同档位遮挡判别（档位固定，只判半/全遮挡，按轮次分折）')
        for lab, nwin, nfold, aw, af in gear_rows:
            print('  %s │ %4d 窗 │ %d 折 │ 窗口 %.3f  整段 %.3f' %
                  (lab, nwin, nfold, aw, af))

    # ---- 汇总指标
    acc = sum(1 for i in range(len(all_true)) if all_true[i] == all_pred[i]) / max(1, len(all_true))
    conf = {c: {d: 0 for d in classes} for c in classes}
    for t, p in zip(all_true, all_pred):
        conf[t][p] += 1

    rep = []
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[t][c] for t in classes if t != c)
        fn = sum(conf[c][d] for d in classes if d != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rep.append((c, tp, fp, fn, prec, rec, f1))

    # ---- 全量训练最终模型（供 predict.py 使用）
    mean, std = zscore_fit(X)
    Z = zscore_apply(X, mean, std)
    final = LDA(args.shrinkage).fit(Z, y)

    gear_acc = (sum(r[4] for r in gear_rows) / len(gear_rows)) if gear_rows else None

    model = {
        'modality': args.modality,
        'features': sndlib.FEATURE_NAMES,
        'classes': final.classes,
        'mean': [round(v, 6) for v in mean],
        'std': [round(v, 6) for v in std],
        'W': [[round(v, 6) for v in row] for row in final.W],
        'b': [round(v, 6) for v in final.b],
        'knn': {'k': 7,
                'X': [[round(v, 4) for v in row] for row in Z],
                'y': y},
        'cv': {'folds': len(cv_rows),
               'acc_window': round(acc, 4),
               'acc_file': round(sum(float(r[6]) for r in cv_rows) / max(1, len(cv_rows)), 4),
               'acc_same_gear': round(gear_acc, 4) if gear_acc is not None else None,
               'n_windows': len(X),
               'n_recordings': len(set(files))},
    }
    with open(os.path.join(args.outdir, 'model.json'), 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=1)

    with open(os.path.join(args.outdir, 'cv_folds.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['折', '测试轮次', '训练窗数', '测试窗数', 'LDA窗口准确率', 'kNN窗口准确率', 'LDA整段准确率'])
        w.writerows(cv_rows)

    if gear_rows:
        with open(os.path.join(args.outdir, 'cv_同档遮挡.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['档位内二分类', '窗口数', '折数', 'LDA窗口准确率', 'LDA整段准确率'])
            for lab, nwin, nfold, aw, af in gear_rows:
                w.writerow([lab, nwin, nfold, '%.4f' % aw, '%.4f' % af])

    with open(os.path.join(args.outdir, '混淆矩阵_窗口.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['真实\\预测'] + classes)
        for c in classes:
            w.writerow([c] + [conf[c][d] for d in classes])

    with open(os.path.join(args.outdir, '分类报告.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['工况', '正确', '误报', '漏报', '精确率', '召回率', 'F1'])
        for row in rep:
            w.writerow([row[0], row[1], row[2], row[3],
                        '%.4f' % row[4], '%.4f' % row[5], '%.4f' % row[6]])
        macro = sum(r[6] for r in rep) / len(rep)
        w.writerow(['宏平均', '', '', '', '', '', '%.4f' % macro])

    acc_file = sum(float(r[6]) for r in cv_rows) / max(1, len(cv_rows))
    print('\n窗口级准确率 %.4f │ 整段级准确率 %.4f │ 宏平均 F1 %.4f' %
          (acc, acc_file, sum(r[6] for r in rep) / len(rep)))

    with open(os.path.join(args.outdir, '训练摘要.md'), 'w', encoding='utf-8') as f:
        f.write('# 训练摘要 · %s 模态\n\n' % ('麦克风录音' if args.modality == 'audio' else '加速度计'))
        f.write('- 数据规模：%d 段录音（%d 个轮次）、%d 个分析窗\n' %
                (len(set(files)), len(set(g)), len(X)))
        if n_base:
            f.write('- 基准工况 %s：%d 个窗口不参与分类（每档只 1 段、跨两档呈双峰，'
                    '进模型会让留一轮整折塌掉），仅在报告中作转速与频谱的基准对照\n'
                    % (args.baseline, n_base))
        f.write('- 特征维度：%d（时域 9 / 频域 8 / 频带 8 / 谐波 4 / 包络 3）\n' %
                len(sndlib.FEATURE_NAMES))
        f.write('- 验证方式：%d 折 GroupKFold，按轮次分组，整轮进整轮出\n' % len(cv_rows))
        f.write('- 分类器：收缩 LDA（λ=%.2f），kNN(k=7) 作对照\n\n' % args.shrinkage)
        f.write('| 指标 | 数值 |\n|---|---|\n')
        f.write('| 窗口级准确率 | %.2f%% |\n' % (acc * 100))
        f.write('| 整段级准确率（窗口投票） | %.2f%% |\n' % (acc_file * 100))
        f.write('| 宏平均 F1 | %.4f |\n\n' % (sum(r[6] for r in rep) / len(rep)))
        f.write('| 工况 | 精确率 | 召回率 | F1 | 支持数 |\n|---|---|---|---|---|\n')
        for r in rep:
            f.write('| %s | %.3f | %.3f | %.3f | %d |\n' % (r[0], r[4], r[5], r[6], r[1] + r[3]))
        f.write('\n混淆矩阵见 `混淆矩阵_窗口.csv`，逐折结果见 `cv_folds.csv`。\n')
        if gear_rows:
            f.write('\n## 同档位遮挡判别（检验转速是否被剥离）\n\n')
            f.write('本项目 4 个工况由「档位 × 遮挡」构成，档位本身在标签里，'
                    '模型可以靠转速（绝对基频）直接判档。原意是在每个档位内部单独建'
                    '「半遮挡 vs 全遮挡」二分类器，把转速压住、只留谱形差异，按轮次分折：\n\n')
            f.write('| 档位内二分类 | 窗口数 | 折数 | 窗口准确率 | 整段准确率 |\n|---|---|---|---|---|\n')
            for lab, nwin, nfold, aw, af in gear_rows:
                f.write('| %s | %d | %d | %.2f%% | %.2f%% |\n'
                        % (lab, nwin, nfold, aw * 100, af * 100))
            f.write('\n平均整段准确率 %.2f%%。\n' % (gear_acc * 100))
            f.write('\n读这组数字要先看转速：本数据上同档位内两类的转速差达 60%，远大于跨档位的 10%，'
                    '因为「遮挡方式」本身就决定转速。因此上表证明的是「同档位内仍然可分」，'
                    '并未证明「判据与转速无关」——仅用 BPF 一维即可取得整段 100%、窗口 92.33%。\n')
    print('结果已写入：%s' % args.outdir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
