# -*- coding: utf-8 -*-
"""
svg_to_png.py —— 把 SVG 转成 2 倍分辨率 PNG，供 Word 报告与手机端查看

用法
    python svg_to_png.py                      # 转 02_可视化图 下全部 SVG
    python svg_to_png.py a.svg b.svg          # 转指定文件
    python svg_to_png.py --scale 3            # 3 倍分辨率

实现方式
    优先调用本机已有的 Edge / Chrome 无头模式截图（Windows 自带 Edge，无需安装任何东西），
    其次尝试 Inkscape、rsvg-convert、ImageMagick。都找不到时给出提示并保留 SVG —
    SVG 本身可被 Word 2019+、浏览器、Inkscape 直接打开，不影响交付。

为什么不用 cairosvg
    它需要 pip 安装与本地 C 库，违背本仓库“零第三方依赖”的约定。
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.abspath(os.path.join(HERE, '..', '02_可视化图'))

BROWSERS = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
]


def find_browser():
    for p in BROWSERS:
        if os.path.isfile(p):
            return p
    for name in ('msedge', 'chrome', 'chromium', 'google-chrome'):
        p = shutil.which(name)
        if p:
            return p
    return None


def find_tool(names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


def svg_size(path):
    """从 <svg width= height= 读出尺寸，失败则退回 960×540。"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            head = f.read(600)
        import re
        m = re.search(r'width="(\d+)"\s+height="(\d+)"', head)
        if m:
            return int(m.group(1)), int(m.group(2))
    except OSError:
        pass
    return 960, 540


def convert(svg, scale, browser=None):
    out = os.path.splitext(svg)[0] + '.png'
    w, h = svg_size(svg)

    if browser:
        url = 'file:///' + os.path.abspath(svg).replace('\\', '/')
        cmd = [browser, '--headless=new', '--disable-gpu', '--hide-scrollbars',
               '--force-device-scale-factor=%d' % scale,
               '--window-size=%d,%d' % (w, h),
               '--default-background-color=FFFFFFFF',
               '--screenshot=' + os.path.abspath(out), url]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.isfile(out):
                return out
        except (OSError, subprocess.SubprocessError):
            pass

    inkscape = find_tool(['inkscape'])
    if inkscape:
        try:
            subprocess.run([inkscape, svg, '--export-type=png', '--export-filename=' + out,
                            '--export-dpi=%d' % (96 * scale)], capture_output=True, timeout=120)
            if os.path.isfile(out):
                return out
        except (OSError, subprocess.SubprocessError):
            pass

    rsvg = find_tool(['rsvg-convert'])
    if rsvg:
        try:
            subprocess.run([rsvg, '--zoom', str(scale), '-o', out, svg],
                           capture_output=True, timeout=120)
            if os.path.isfile(out):
                return out
        except (OSError, subprocess.SubprocessError):
            pass

    magick = find_tool(['magick', 'convert'])
    if magick:
        try:
            subprocess.run([magick, '-density', str(96 * scale), svg, out],
                           capture_output=True, timeout=120)
            if os.path.isfile(out):
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description='SVG → PNG')
    ap.add_argument('files', nargs='*')
    ap.add_argument('--scale', type=int, default=2)
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(FIG_DIR, '*.svg')))
    if not files:
        print('没有找到 SVG，先运行 python plot_figs.py')
        return 2

    browser = find_browser()
    print('渲染器：%s' % (browser or '未找到浏览器/转换工具'))

    ok, fail = 0, []
    for svg in files:
        out = convert(svg, args.scale, browser)
        if out:
            size = os.path.getsize(out) / 1024.0
            print('  %s -> %s (%.0f KB)' % (os.path.basename(svg), os.path.basename(out), size))
            ok += 1
        else:
            fail.append(os.path.basename(svg))

    if fail:
        print('\n%d 张未能转换：%s' % (len(fail), '、'.join(fail)))
        print('提示：本机没有可用的渲染器时，SVG 仍可用浏览器打开、或用 Word「插入图片」直接选 SVG，'
              '不影响交付；需要 PNG 时用 Edge 打开 SVG 后截图亦可。')
    print('\n完成 %d / %d' % (ok, len(files)))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
