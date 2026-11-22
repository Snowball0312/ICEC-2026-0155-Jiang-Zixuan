# -*- coding: utf-8 -*-
"""
plotsvg.py —— 极简 SVG 绘图库（纯标准库，零第三方依赖）

只提供画图表真正需要的几样东西：坐标轴、刻度、折线、填充面积、热力格、图例、文字标注。
输出为标准 SVG 1.1，可用浏览器 / Word / Inkscape / Illustrator 打开，也可用 svg_to_png.py 转 PNG。

设计取舍
    不追求通用绘图能力，只服务本项目 6 张图；颜色、字号、边距集中在文件顶部，
    改一处全图生效，避免每张图各写一套样式。
"""

import math

FONT = '"Microsoft YaHei","PingFang SC","Hiragino Sans GB",sans-serif'
MONO = 'ui-monospace,Consolas,"Cascadia Mono",monospace'

# 五种工况的固定配色，与交付清单、报告保持一致
COLORS = ['#0F6E56', '#378ADD', '#BA7517', '#D85A30', '#534AB7']
GRID = '#E3E6EC'
AXIS = '#9AA1AE'
INK = '#16181D'
INK2 = '#4A4F5A'
INK3 = '#7C8493'


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def nice_ticks(lo, hi, n=6):
    """生成好看的刻度值（1/2/5 × 10^k 步长）。"""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(1, n - 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if (hi - lo) / step <= n:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            ticks.append(round(v, 10))
        v += step
    return ticks


class Chart:
    """一张图的画布。坐标系由 xlim / ylim 决定，绘制方法负责映射到像素。"""

    def __init__(self, width=960, height=540, title='', xlabel='', ylabel='',
                 xlim=(0, 1), ylim=(0, 1), pad=(78, 34, 54, 64), log_y=False):
        self.w, self.h = width, height
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.xlim, self.ylim = xlim, ylim
        self.pl, self.pr, self.pt, self.pb = pad
        self.log_y = log_y
        self.body = []
        self.plot = []
        self.legends = []

    # ---- 坐标映射
    def px(self, x):
        lo, hi = self.xlim
        return self.pl + (x - lo) / (hi - lo or 1) * (self.w - self.pl - self.pr)

    def py(self, y):
        lo, hi = self.ylim
        if self.log_y:
            lo = max(lo, 1e-12)
            y = max(y, 1e-12)
            t = (math.log10(y) - math.log10(lo)) / (math.log10(hi) - math.log10(lo) or 1)
        else:
            t = (y - lo) / (hi - lo or 1)
        return self.h - self.pb - t * (self.h - self.pt - self.pb)

    # ---- 装配
    def _ticks_x(self):
        ticks = nice_ticks(self.xlim[0], self.xlim[1], 6)
        out = []
        for t in ticks:
            x = self.px(t)
            out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="0.5"/>'
                       % (x, self.pt, x, self.h - self.pb, GRID))
            out.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="middle">%s</text>'
                       % (x, self.h - self.pb + 19, INK3, _fmt(t)))
        return out

    def _ticks_y(self, vals=None, n=6):
        out = []
        lo, hi = self.ylim
        if vals is not None:
            self._yticks = list(vals)
            for t in vals:
                y = self.py(t)
                out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="0.5"/>'
                           % (self.pl, y, self.w - self.pr, y, GRID))
                out.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end" '
                           'dominant-baseline="central">%s</text>'
                           % (self.pl - 9, y, INK3, _fmt(t)))
            return out
        if self.log_y:
            vals = []
            e = math.floor(math.log10(max(lo, 1e-12)))
            while 10 ** e <= hi * 1.01:
                if 10 ** e >= lo * 0.99:
                    vals.append(10 ** e)
                if len(vals) > 8:
                    break
                e += 1
        else:
            vals = nice_ticks(lo, hi, n)
        self._yticks = vals
        for t in vals:
            y = self.py(t)
            out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="%s" stroke-width="0.5"/>'
                       % (self.pl, y, self.w - self.pr, y, GRID))
            out.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end" '
                       'dominant-baseline="central">%s</text>'
                       % (self.pl - 9, y, INK3, _fmt(t, log=self.log_y)))
        return out

    def axes(self, yticks='auto', xticks='auto'):
        if xticks != 'none':
            self.body += self._ticks_x()
        if yticks != 'none':
            self.body += self._ticks_y(yticks if isinstance(yticks, (list, tuple)) else None)
        # 轴线
        self.body.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
                         % (self.pl, self.pt, self.pl, self.h - self.pb, AXIS))
        self.body.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1"/>'
                         % (self.pl, self.h - self.pb, self.w - self.pr, self.h - self.pb, AXIS))
        if self.xlabel:
            self.body.append('<text x="%.1f" y="%d" font-size="12.5" fill="%s" text-anchor="middle">%s</text>'
                             % ((self.pl + self.w - self.pr) / 2, self.h - 16, INK2, esc(self.xlabel)))
        if self.ylabel:
            self.body.append('<text x="%d" y="%.1f" font-size="12.5" fill="%s" text-anchor="middle" '
                             'transform="rotate(-90 %d %.1f)">%s</text>'
                             % (20, (self.pt + self.h - self.pb) / 2, INK2, 20,
                                (self.pt + self.h - self.pb) / 2, esc(self.ylabel)))
        if self.title:
            self.body.append('<text x="%d" y="26" font-size="15" font-weight="500" fill="%s">%s</text>'
                             % (self.pl, INK, esc(self.title)))

    # ---- 图形
    def line(self, xs, ys, color=COLORS[0], width=1.6, label=None, dash=None, opacity=1.0):
        pts = []
        for x, y in zip(xs, ys):
            pts.append('%.1f,%.1f' % (self.px(x), self.py(y)))
        d = ' stroke-dasharray="%s"' % dash if dash else ''
        self.plot.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="%.1f" '
                         'stroke-linejoin="round" stroke-linecap="round" opacity="%.2f"%s/>'
                         % (' '.join(pts), color, width, opacity, d))
        if label:
            self.legends.append((label, color))

    def area(self, xs, ys, color=COLORS[0], opacity=0.16):
        if not xs:
            return
        pts = ['%.1f,%.1f' % (self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        base = '%.1f,%.1f' % (self.px(xs[-1]), self.h - self.pb)
        head = '%.1f,%.1f' % (self.px(xs[0]), self.h - self.pb)
        self.plot.append('<polygon points="%s %s %s" fill="%s" opacity="%.2f" stroke="none"/>'
                         % (head, ' '.join(pts), base, color, opacity))

    def bars(self, xs, ys, colors=None, width=None, opacity=0.9):
        n = len(xs) or 1
        if width is None:
            width = max(2.0, (self.w - self.pl - self.pr) / max(1, n) * 0.62)
        for i, (x, y) in enumerate(zip(xs, ys)):
            cx = self.px(x)
            y0 = self.py(self.ylim[0])
            y1 = self.py(y)
            c = colors[i] if colors else COLORS[0]
            self.plot.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" '
                             'opacity="%.2f" rx="2"/>'
                             % (cx - width / 2, min(y0, y1), width, abs(y0 - y1), c, opacity))

    def heat(self, matrix, row_labels, col_labels, vmax=None):
        """热力格（混淆矩阵），格内写数值，颜色越深值越大。"""
        n, m = len(matrix), len(matrix[0])
        vmax = vmax or max(max(r) for r in matrix) or 1
        cw = (self.w - self.pl - self.pr) / m
        ch = (self.h - self.pt - self.pb) / n
        for i in range(n):
            for j in range(m):
                v = matrix[i][j]
                t = v / vmax
                r, g, b = _mix((250, 251, 252), (15, 110, 86), t * 0.85)
                x = self.pl + j * cw
                y = self.pt + i * ch
                self.body.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="rgb(%d,%d,%d)" '
                                 'stroke="#ffffff" stroke-width="1.5"/>' % (x, y, cw, ch, r, g, b))
                self.body.append('<text x="%.1f" y="%.1f" font-size="13" fill="%s" text-anchor="middle" '
                                 'dominant-baseline="central">%d</text>'
                                 % (x + cw / 2, y + ch / 2, '#FFFFFF' if t > 0.55 else INK, v))
        for i, lab in enumerate(row_labels):
            self.body.append('<text x="%d" y="%.1f" font-size="12" fill="%s" text-anchor="end" '
                             'dominant-baseline="central">%s</text>'
                             % (self.pl - 10, self.pt + i * ch + ch / 2, INK2, esc(lab)))
        for j, lab in enumerate(col_labels):
            self.body.append('<text x="%.1f" y="%d" font-size="12" fill="%s" text-anchor="end" '
                             'transform="rotate(-38 %.1f %d)">%s</text>'
                             % (self.pl + j * cw + cw / 2, self.h - self.pb + 18, INK2,
                                self.pl + j * cw + cw / 2, self.h - self.pb + 18, esc(lab)))

    def legend(self, x=None, y=None):
        if not self.legends:
            return
        x = self.w - self.pr - 150 if x is None else x
        y = self.pt + 12 if y is None else y
        for i, (lab, color) in enumerate(self.legends):
            yy = y + i * 19
            self.body.append('<rect x="%d" y="%d" width="18" height="3" fill="%s" rx="1.5"/>'
                             % (x, yy, color))
            self.body.append('<text x="%d" y="%d" font-size="12" fill="%s" dominant-baseline="central">%s</text>'
                             % (x + 25, yy + 1, INK2, esc(lab)))

    def note(self, text, x=None, y=None, anchor='start', color=INK3, size=11.5):
        x = self.pl + 8 if x is None else x
        y = 47 if y is None else y
        self.body.append('<text x="%.1f" y="%.1f" font-size="%s" fill="%s" text-anchor="%s">%s</text>'
                         % (x, y, size, color, anchor, esc(text)))

    def save(self, path):
        head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                'viewBox="0 0 %d %d">\n'
                '<rect width="100%%" height="100%%" fill="#ffffff"/>\n'
                '<defs><clipPath id="plotclip"><rect x="%d" y="%d" width="%d" height="%d"/>'
                '</clipPath></defs>\n'
                '<g font-family=\'%s\'>\n'
                % (self.w, self.h, self.w, self.h, self.pl, self.pt,
                   self.w - self.pl - self.pr, self.h - self.pt - self.pb, FONT))
        tail = '</g>\n</svg>\n'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(head + '\n'.join(self.body) +
                    '\n<g clip-path="url(#plotclip)">\n' + '\n'.join(self.plot) +
                    '\n</g>\n' + tail)
        return path


def _fmt(v, log=False):
    if log:
        if v >= 1000:
            return '%gk' % (v / 1000.0)
        if v >= 1:
            return '%g' % v
        return ('%.2f' % v).rstrip('0').rstrip('.')
    if abs(v) >= 1000:
        return '%gk' % round(v / 1000.0, 1)
    if abs(v) >= 10:
        return '%g' % round(v, 1)
    if abs(v) >= 1:
        return '%g' % round(v, 2)
    return '%g' % round(v, 3)


def _mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))
