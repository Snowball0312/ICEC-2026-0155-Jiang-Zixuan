# -*- coding: utf-8 -*-
"""
AI 组耗时实测（纯改名口径）：对 10 组样本各跑 3 轮
==================================================
测量口径 = 端到端（含 Python 解释器启动 + 读文件解析 + 原地改名落盘），
即"把脚本丢进文件夹、按下回车到全部改完"的真实体验耗时。

输出 results/ai_timing.csv（逐轮）+ results/ai_timing_summary.csv（每组中位数）
"""
from __future__ import annotations

import csv
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SAMPLES = BASE / "samples_raw"
WORK = BASE / "runs" / "timing"
RESULTS = BASE / "results"
SCRIPT = BASE / "scripts" / "batch_rename.py"
REPS = 3


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    rows = []
    for rep in range(1, REPS + 1):
        for g in sorted(p.name for p in SAMPLES.iterdir() if p.is_dir()):
            src = WORK / g
            if src.exists():
                shutil.rmtree(src)
            shutil.copytree(SAMPLES / g, src)

            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), str(src), "--log", str(WORK / f"{g}_日志"), "--quiet"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            elapsed = time.perf_counter() - t0
            if proc.returncode != 0:
                print("运行失败:", proc.stderr)
                return

            n = len([p for p in src.iterdir() if p.is_file()])
            rows.append({"rep": rep, "group": g, "files": n, "elapsed_s": f"{elapsed:.4f}"})
            print(f"第{rep}轮 {g}: {elapsed:.4f}s")

    with (RESULTS / "ai_timing.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "group", "files", "elapsed_s"])
        w.writeheader()
        w.writerows(rows)

    summary = []
    for g in sorted({r["group"] for r in rows}):
        vals = [float(r["elapsed_s"]) for r in rows if r["group"] == g]
        summary.append(
            {
                "group": g,
                "files": 12,
                "runs": " / ".join(f"{v:.4f}" for v in vals),
                "min_s": f"{min(vals):.4f}",
                "median_s": f"{statistics.median(vals):.4f}",
                "max_s": f"{max(vals):.4f}",
            }
        )
    with (RESULTS / "ai_timing_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "files", "runs", "min_s", "median_s", "max_s"])
        w.writeheader()
        w.writerows(summary)

    all_med = [float(r["median_s"]) for r in summary]
    print(f"\n单组中位数均值 {statistics.mean(all_med):.4f}s，总 {sum(all_med):.4f}s")


if __name__ == "__main__":
    main()
