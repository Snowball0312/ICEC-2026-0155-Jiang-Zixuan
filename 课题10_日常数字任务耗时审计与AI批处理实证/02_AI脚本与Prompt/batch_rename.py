# -*- coding: utf-8 -*-
"""
AI 批处理脚本：报销票据批量改名（纯改名，不建文件夹、不移动）
==========================================================
规则：
  · 票据文件 → 改名为  YYYYMMDD_类别_金额.txt  （原地改名）
  · 非票据文件（读不出日期/类别/金额）→ 保持原名不动
  · 已是标准名的文件 → 跳过（重复运行不产生任何变化，保证幂等）

输出：日志目录下的 manifest.csv，逐文件记录处理结果。

用法：
  python batch_rename.py <收件夹路径> [--log 日志目录]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

TYPES = ("餐饮", "交通", "办公", "住宿", "其他")

RE_DASH_DATE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})")
RE_COMPACT_DATE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
RE_AMOUNT = re.compile(r"(?:金额|价税合计|合计|总额)[\s:：]*[¥￥]?\s*([0-9]+(?:\.[0-9]+)?)")
RE_PIPE = re.compile(r"^\s*(\d{8})\s*\|\s*(.+?)\s*\|\s*([0-9.]+)")
RE_STD_NAME = re.compile(r"^\d{8}_(餐饮|交通|办公|住宿|其他)_\d+\.\d{2}\.txt$")


def read_text(p: Path) -> str:
    """按 utf-8-sig → gbk 依次尝试解码，兼容 BOM 与 GBK 文件"""
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def normalize(s: str) -> str:
    """全角空格、全角冒号、全角逗号统一为半角，换行统一为 \\n"""
    return (
        s.replace("\u3000", " ")
        .replace("：", ":")
        .replace("，", ",")
        .replace("\r\n", "\n")
    )


def parse_date(text: str) -> str | None:
    m = RE_DASH_DATE.search(text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}{mo:02d}{d:02d}"
        return None
    m = RE_COMPACT_DATE.search(text)
    if m:
        return m.group(0)
    return None


def parse_type(text: str) -> str | None:
    # 优先取标签行（费用类型 / 类别），避免正文"其他"字样造成误判
    for line in text.split("\n"):
        if re.search(r"(费用类型|类别|类型|科目)", line):
            for t in TYPES:
                if t in line:
                    return t
    m = RE_PIPE.search(text)
    if m:
        field = m.group(2)
        for t in TYPES:
            if t in field:
                return t
    for t in TYPES:
        if t in text:
            return t
    return None


def parse_amount(text: str) -> str | None:
    m = RE_AMOUNT.search(text)
    if not m:
        m = RE_PIPE.search(text)
        if m:
            return f"{float(m.group(3)):.2f}"
        cands = re.findall(r"[¥￥]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not cands:
            return None
        return f"{float(cands[0]):.2f}"
    return f"{float(m.group(1)):.2f}"


def process(src: Path, log_dir: Path, quiet: bool = False) -> dict:
    """处理一个收件夹：原地改名，返回统计信息"""
    files = sorted([p for p in src.iterdir() if p.is_file()], key=lambda p: p.name)
    rows: list[dict] = []
    renamed = skipped = untouched = 0

    for f in files:
        text = normalize(read_text(f))
        d = parse_date(text)
        t = parse_type(text)
        a = parse_amount(text)

        # 非票据：保持原名不动
        if not (d and t and a):
            untouched += 1
            rows.append(
                {
                    "原文件名": f.name,
                    "新文件名": f.name,
                    "状态": "非票据·保持原名",
                    "日期": d or "",
                    "类型": t or "",
                    "金额": a or "",
                    "说明": "读不出完整票据字段",
                }
            )
            continue

        new_name = f"{d}_{t}_{a}.txt"

        # 已是标准名：跳过，保证重复运行无副作用
        if RE_STD_NAME.match(f.name) and f.name == new_name:
            skipped += 1
            rows.append(
                {
                    "原文件名": f.name,
                    "新文件名": new_name,
                    "状态": "已是标准名·跳过",
                    "日期": d,
                    "类型": t,
                    "金额": a,
                    "说明": "",
                }
            )
            continue

        # 同名冲突（不同内容撞名）：加序号，绝不覆盖
        target = src / new_name
        if target.exists() and target != f:
            base = f"{d}_{t}_{a}"
            k = 2
            while (src / f"{base}_{k}.txt").exists():
                k += 1
            new_name = f"{base}_{k}.txt"
            target = src / new_name

        f.rename(target)
        renamed += 1
        rows.append(
            {
                "原文件名": f.name,
                "新文件名": new_name,
                "状态": "已改名",
                "日期": d,
                "类型": t,
                "金额": a,
                "说明": "",
            }
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    mf = log_dir / f"{src.name}_manifest.csv"
    with mf.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["原文件名"])
        w.writeheader()
        w.writerows(rows)

    stat = {"files": len(files), "renamed": renamed, "skipped": skipped, "untouched": untouched}
    if not quiet:
        print(
            f"[完成] 共 {stat['files']} 份 → 改名 {renamed} / 跳过 {skipped} / 非票据 {untouched}"
        )
    return stat


def main() -> int:
    ap = argparse.ArgumentParser(description="报销票据批量改名（纯改名，不移动）")
    ap.add_argument("src", help="收件夹路径")
    ap.add_argument("--log", default=None, help="日志目录（默认在收件夹同级生成 <名称>_日志）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"路径不存在：{src}", file=sys.stderr)
        return 2
    log_dir = Path(args.log).resolve() if args.log else src.parent / f"{src.name}_日志"

    t0 = time.perf_counter()
    stat = process(src, log_dir, quiet=args.quiet)
    elapsed = time.perf_counter() - t0
    print(f"[耗时] {elapsed:.4f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
