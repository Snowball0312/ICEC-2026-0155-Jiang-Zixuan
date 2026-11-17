# -*- coding: utf-8 -*-
"""
AI 组正确性核验（纯改名口径）
==========================
直接比对「处理后的目录实际文件名」与「标准答案要求的新文件名」：

  1. 票据：目录里必须出现 YYYYMMDD_类别_金额.txt
  2. 非票据：必须仍是原名（不被误改名）
  3. 一份不多、一份不少（文件总数不变，无丢失、无覆盖）

输出 results/ai_verify.csv（每组正确/错误数）与 results/ai_errors.csv（错误明细）
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRUTH = BASE / "ground_truth"
RESULTS = BASE / "results"
RUNS = BASE / "runs" / "timing"

RE_SUFFIX = re.compile(r"^(.+)_(\d+)\.txt$")  # 撞名加序号：base_2.txt


def base_of(name: str) -> str:
    """把 base_2.txt 还原成 base.txt，便于容忍撞名序号"""
    m = RE_SUFFIX.match(name)
    if m and re.match(r"^\d{8}_", m.group(1)):
        return m.group(1) + ".txt"
    return name


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    summary, errors = [], []

    for gt in sorted(TRUTH.glob("sample-*.csv")):
        g = gt.stem
        run_dir = RUNS / g
        if not run_dir.is_dir():
            print(f"缺少运行目录：{run_dir}")
            continue

        with gt.open(encoding="utf-8-sig") as f:
            truth = list(csv.DictReader(f))

        expected = {r["expected_name"] for r in truth}
        actual_raw = {p.name for p in run_dir.iterdir() if p.is_file()}
        actual = {base_of(n) for n in actual_raw}

        hit = expected & actual
        miss = expected - actual       # 该出现的名字没出现
        extra = actual - expected      # 出现了不该有的名字

        correct = len(hit)
        total = len(expected)
        for nm in sorted(miss):
            errors.append({"group": g, "filename": nm, "reason": "应有此名但未出现（漏改或改错）"})
        for nm in sorted(extra):
            errors.append({"group": g, "filename": nm, "reason": "出现了不该有的名字（误改名）"})
        if len(actual_raw) != len(truth):
            errors.append(
                {
                    "group": g,
                    "filename": f"文件总数 {len(actual_raw)} ≠ 应有 {len(truth)}",
                    "reason": "数量不符（丢失或覆盖）",
                }
            )

        summary.append(
            {
                "group": g,
                "total": total,
                "correct": correct,
                "wrong": total - correct,
                "accuracy": f"{correct / total:.4f}" if total else "0",
            }
        )

    with (RESULTS / "ai_verify.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "total", "correct", "wrong", "accuracy"])
        w.writeheader()
        w.writerows(summary)
    with (RESULTS / "ai_errors.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "filename", "reason"])
        w.writeheader()
        w.writerows(errors)

    tot = sum(r["total"] for r in summary)
    cor = sum(r["correct"] for r in summary)
    if tot:
        print(f"核验完成：{cor}/{tot} 正确，准确率 {cor / tot:.2%}，异常 {len(errors)} 条")
    for r in summary:
        print(f"  {r['group']}: {r['correct']}/{r['total']}  {r['accuracy']}")


if __name__ == "__main__":
    main()
