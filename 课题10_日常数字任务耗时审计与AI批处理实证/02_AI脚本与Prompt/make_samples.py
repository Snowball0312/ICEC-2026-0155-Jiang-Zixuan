# -*- coding: utf-8 -*-
"""
课题10 模拟样本生成器（简化版：纯改名）
--------------------------------------
生成 10 组「报销票据收件夹」，每组 12 个纯文本文件：
  · 10 份票据（文件名混乱、字段格式 4 种变体）
  · 2 份非票据文件（扫描空白页、行政通知）→ 规则要求保持原名不动

任务只有一件事：把票据文件改名成 YYYYMMDD_类别_金额.txt，不建文件夹、不移动。

同时输出 ground_truth/，记录每份文件的正确新文件名，用于核算准确率。

运行： python make_samples.py
"""
from __future__ import annotations

import csv
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SAMPLES = BASE / "samples_raw"
TRUTH = BASE / "ground_truth"

TYPES = ["餐饮", "交通", "办公", "住宿", "其他"]
AMOUNT_RANGE = {
    "餐饮": (28, 480),
    "交通": (12, 260),
    "办公": (35, 900),
    "住宿": (180, 1200),
    "其他": (20, 600),
}
NOTE_POOL = ["项目组出差", "客户接待", "部门月度采购", "市场活动", "季度例会", "", "", "补开发票"]

# 混乱文件名模板
NAME_TPL = [
    "IMG_{ymd}_{i:03d}.txt",
    "扫描件 ({i}).txt",
    "WeChat Image_{ymd}{hms}.txt",
    "扫描全能王 {ymd} {hms}.txt",
    "New Text Document ({i}).txt",
    "票据{i}.TXT",
    "Screenshot_{ymd}-{hms}.txt",
    "来自 iPhone {i}.txt",
    "扫描件副本 ({i}).txt",
    "¥临时文件{i}.txt",
]


def fmt_amt(v: float) -> str:
    return f"{v:.2f}"


def render(style: int, d: date, t: str, amt: float, rid: str, note: str) -> str:
    """4 种字段格式变体，模拟真实收件夹里的格式不一致"""
    y, m, dd = d.year, d.month, d.day
    ymd = f"{y}{m:02d}{dd:02d}"
    if style == 0:  # 标准冒号 + 短横日期 + ￥
        return (
            f"单据编号：{rid}\n"
            f"开票日期：{y}-{m:02d}-{dd:02d}\n"
            f"费用类型：{t}\n"
            f"金额：¥{fmt_amt(amt)}\n"
            f"备注：{note}\n"
        )
    if style == 1:  # 方括号 + 斜杠日期 + 元 + 金额不带两位小数
        raw = f"{amt:.2f}".rstrip("0").rstrip(".")
        return (
            "[票据信息]\n"
            f"日期 {y}/{m}/{dd}\n"
            f"类别 {t}\n"
            f"价税合计 {raw} 元\n"
            f"编号 {rid}\n"
        )
    if style == 2:  # 管道分隔单行
        return f"{ymd} | {t} | {fmt_amt(amt)} | {rid}\n"
    # 全角空格 + 中文年月日
    return (
        f"开票日期：{y}年{m}月{dd}日\n"
        f"费用类型：\u3000{t}\n"
        f"金额：\u3000{fmt_amt(amt)} 元\n"
        f"单据编号：{rid}\n"
    )


def build_group(idx: int) -> None:
    rng = random.Random(20260900 + idx)
    out_dir = SAMPLES / f"sample-{idx:02d}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    start = date(2026, 8, 1) + timedelta(days=rng.randint(0, 40))
    used_names: set[str] = set()
    rows: list[dict] = []
    used_key: set[tuple] = set()

    # 每组固定 10 份票据 + 2 份非票据 = 12 份，保证样本量一致
    n_ticket = 10
    for _ in range(n_ticket):
        # 保证 (日期, 类型, 金额) 组合唯一，避免重名
        for _ in range(200):
            d = start + timedelta(days=rng.randint(0, 75))
            t = rng.choice(TYPES)
            lo, hi = AMOUNT_RANGE[t]
            amt = round(rng.uniform(lo, hi), 2)
            if (d, t, amt) not in used_key:
                used_key.add((d, t, amt))
                break
        rid = f"FP-{d.year}{d.month:02d}{d.day:02d}-{rng.randint(1000, 9999)}"
        note = rng.choice(NOTE_POOL)
        style = rng.choice([0, 0, 1, 1, 2, 3])

        tpl = rng.choice(NAME_TPL)
        name = tpl.format(
            ymd=f"{d.year}{d.month:02d}{d.day:02d}",
            i=rng.randint(1, 99),
            hms=f"{rng.randint(8, 21):02d}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}",
        )
        while name in used_names:
            name = f"{Path(name).stem} ({rng.randint(2, 99)}).txt"
        used_names.add(name)

        body = render(style, d, t, amt, rid, note)
        # 制造轻微"脏数据"：部分文件带 BOM、CRLF、行首空行
        mode = rng.choice(["plain", "bom", "crlf", "blank"])
        if mode == "blank":
            body = "\n\n" + body
        if mode == "bom":
            (out_dir / name).write_text(body, encoding="utf-8-sig", newline="")
        elif mode == "crlf":
            (out_dir / name).write_text(body.replace("\n", "\r\n"), encoding="utf-8", newline="")
        else:
            (out_dir / name).write_text(body, encoding="utf-8", newline="")

        rows.append(
            {
                "group": f"sample-{idx:02d}",
                "filename": name,
                "kind": "票据",
                "date": d.isoformat(),
                "type": t,
                "amount": fmt_amt(amt),
                "receipt_id": rid,
                "expected_name": f"{d.year}{d.month:02d}{d.day:02d}_{t}_{fmt_amt(amt)}.txt",
            }
        )

    # 非票据文件：规则要求保持原名不动
    noises = [
        ("扫描失败_空白页.txt", "本页扫描失败，图像空白，未见有效票据信息。请重新扫描后上传。\n"),
        ("README.txt", "本目录为 2026 年度报销收件夹。\n票据请按 YYYYMMDD_类别_金额 重命名。\n—— 行政部\n"),
    ]
    for nm, txt in noises:
        if (out_dir / nm).exists():
            continue
        (out_dir / nm).write_text(txt, encoding="utf-8", newline="")
        rows.append(
            {
                "group": f"sample-{idx:02d}",
                "filename": nm,
                "kind": "非票据",
                "date": "",
                "type": "",
                "amount": "",
                "receipt_id": "",
                "expected_name": nm,  # 保持原名
            }
        )

    rows.sort(key=lambda r: r["filename"])
    TRUTH.mkdir(exist_ok=True)
    with (TRUTH / f"{out_dir.name}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    for i in range(1, 11):
        build_group(i)
    print("已生成 10 组样本（纯改名任务）")
    for i in range(1, 11):
        d = SAMPLES / f"sample-{i:02d}"
        n_t = sum(1 for r in csv.DictReader((TRUTH / f"{d.name}.csv").open(encoding="utf-8-sig")) if r["kind"] == "票据")
        print(f"  {d.name}: {len(list(d.iterdir()))} 个文件（其中票据 {n_t} 份）")


if __name__ == "__main__":
    main()
