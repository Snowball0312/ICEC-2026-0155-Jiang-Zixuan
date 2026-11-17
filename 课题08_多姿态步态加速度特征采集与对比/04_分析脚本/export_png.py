"""把 outputs/ 下的 SVG 转成 PNG。

SVG 适合矢量印刷，但贴进 Word、PPT 或聊天窗口时常出现字体回退、方框或缩放模糊，
因此同时提供 PNG 版本。转换借用本机已安装的 Chromium 内核浏览器（Edge / Chrome）
的无头截图能力，不引入任何第三方 Python 依赖。

用法：
    python export_png.py            # 转换 outputs/ 下全部 SVG
    python export_png.py --scale 2  # 输出 2 倍分辨率，便于投影或印刷
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def find_browser() -> str | None:
    for name in ("msedge", "chrome", "chromium", "google-chrome"):
        located = shutil.which(name)
        if located:
            return located
    for path in CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def svg_size(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")[:2000]
    width = re.search(r'width="(\d+)"', text)
    height = re.search(r'height="(\d+)"', text)
    if not width or not height:
        raise ValueError(f"无法从 {path.name} 的头部解析画布尺寸")
    return int(width.group(1)), int(height.group(1))


def convert(browser: str, svg_path: Path, scale: float) -> Path:
    width, height = svg_size(svg_path)
    png_path = svg_path.with_suffix(".png")
    with tempfile.TemporaryDirectory() as profile:
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            f"--force-device-scale-factor={scale}",
            f"--window-size={width},{height}",
            f"--screenshot={png_path}",
            svg_path.resolve().as_uri(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if not png_path.is_file() or png_path.stat().st_size == 0:
        message = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise RuntimeError(f"转换 {svg_path.name} 失败：{message[-1] if message else '无输出'}")
    return png_path


def main() -> int:
    parser = argparse.ArgumentParser(description="将 outputs/ 下的 SVG 批量转换为 PNG")
    parser.add_argument("--scale", type=float, default=2.0, help="输出倍率，默认 2 倍")
    arguments = parser.parse_args()

    browser = find_browser()
    if not browser:
        print("未找到 Edge / Chrome / Chromium。", file=sys.stderr)
        print("请安装其中任意一个，或手动用浏览器打开 SVG 后另存为 PNG。", file=sys.stderr)
        return 1
    print(f"使用浏览器：{browser}")

    sources = sorted(OUTPUT_DIR.glob("*.svg"))
    if not sources:
        print(f"{OUTPUT_DIR} 下没有 SVG 文件，请先运行 analyze_acceleration.py", file=sys.stderr)
        return 1

    failures = 0
    for source in sources:
        try:
            target = convert(browser, source, arguments.scale)
            print(f"  {source.name} → {target.name}（{target.stat().st_size // 1024} KB）")
        except Exception as error:  # noqa: BLE001 - 逐文件报错，不中断其余转换
            failures += 1
            print(f"  {source.name} 失败：{error}", file=sys.stderr)

    print(f"完成 {len(sources) - failures}/{len(sources)} 个文件。")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
