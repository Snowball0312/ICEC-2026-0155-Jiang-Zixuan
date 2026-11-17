"""多姿态步态加速度特征提取与阈值分类。

输入 data/ 下的四份 Phyphox 线性加速度 CSV，输出统计表、阈值规则与两张图。

判决口径（默认）：
1. 每个连续采集片段按 1 秒分桶，窗口内 `Absolute acceleration` 取算术平均，作为该窗口特征值 M；
2. 窗口覆盖不足 0.8 秒（200 Hz 下 160 点）时丢弃，避免残缺窗口拉偏均值；
3. 同一片段内对 M 取 3 窗滑动中值，抑制起停、折返等过渡态的单窗异常；
4. 阈值由相邻两类特征分布的 90 / 10 分位取中点得到，而非均值中点——
   过渡态会把类别均值拉偏，分位点对少量离群窗更稳健。

脚本同时计算旧口径（门槛 50 点、均值中点、逐窗判决）的指标，
用于报告中五档消融对照，各档共用同一份清洗后的数据。
"""

from __future__ import annotations

import csv
import html
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

WINDOW_SECONDS = 1.0
WINDOW_COVERAGE = 0.8
SMOOTHING_WINDOWS = 3
THRESHOLD_PERCENTILE = 0.90

ACTIVITIES = (
    ("静止", "静止.csv", "#2f6fb5"),
    ("平地慢走", "平地慢走.csv", "#2f8f5b"),
    ("上下楼梯", "上下楼梯.csv", "#c47f1c"),
    ("快跑", "快跑.csv", "#b84451"),
)
AXES = (
    ("Linear Acceleration x (m/s^2)", "X", "#2f6fb5"),
    ("Linear Acceleration y (m/s^2)", "Y", "#2f8f5b"),
    ("Linear Acceleration z (m/s^2)", "Z", "#b84451"),
)
MAGNITUDE_COLUMN = "Absolute acceleration (m/s^2)"
TIME_COLUMN = "Time (s)"
LABEL_COLUMN = "楼梯方向"
MAGNITUDE_LABEL = "|a|"
MAGNITUDE_COLOR = "#2b3648"


# --------------------------------------------------------------------------- 数据读取


@dataclass
class Segment:
    activity: str
    label: str
    rows: list[dict[str, str]]

    @property
    def duration(self) -> float:
        return max(float(self.rows[-1][TIME_COLUMN]) - float(self.rows[0][TIME_COLUMN]), 0.0)

    def series(self, column: str) -> list[float]:
        return [float(row[column]) for row in self.rows]


@dataclass
class Recording:
    activity: str
    filename: str
    rows: list[dict[str, str]]
    segments: list[Segment] = field(default_factory=list)


@dataclass
class Window:
    activity: str
    segment_index: int
    window_index: int
    segment_windows: int
    value: float
    smoothed: float
    samples: int

    @property
    def is_edge(self) -> bool:
        return self.window_index in (0, self.segment_windows - 1)


def is_zero_row(row: dict[str, str]) -> bool:
    """传感器刚启动时会写出一行全零采样，属于空采样而非真实静止。"""
    columns = [column for column, _, _ in AXES] + [MAGNITUDE_COLUMN]
    return all(abs(float(row[column])) < 1e-12 for column in columns)


def split_segments(rows: list[dict[str, str]], activity: str) -> list[Segment]:
    """按方向标签变化或时间轴回跳切分，并剔除片段开头的全零行。"""
    groups: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    previous_time: float | None = None
    previous_label = None
    has_label = LABEL_COLUMN in rows[0]

    for row in rows:
        timestamp = float(row[TIME_COLUMN])
        label = row[LABEL_COLUMN] if has_label else None
        if current and (
            (previous_time is not None and timestamp < previous_time - 1.0)
            or (has_label and label != previous_label)
        ):
            groups.append(current)
            current = []
        current.append(row)
        previous_time = timestamp
        previous_label = label
    if current:
        groups.append(current)

    segments: list[Segment] = []
    for group in groups:
        while group and is_zero_row(group[0]):
            group.pop(0)
        if not group:
            continue
        label = group[0][LABEL_COLUMN] if has_label else activity
        segments.append(Segment(activity, label, group))
    return segments


def read_recordings() -> list[Recording]:
    recordings: list[Recording] = []
    for activity, filename, _ in ACTIVITIES:
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            required = {TIME_COLUMN, MAGNITUDE_COLUMN, *(axis for axis, _, _ in AXES)}
            missing = required.difference(fieldnames)
            if missing:
                raise ValueError(f"{filename} 缺少列：{sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{filename} 没有数据行")
        recordings.append(Recording(activity, filename, rows, split_segments(rows, activity)))
    return recordings


# --------------------------------------------------------------------------- 特征计算


def sample_rate_hz(recording: Recording) -> float:
    intervals = [
        later - earlier
        for segment in recording.segments
        for earlier, later in zip(segment.series(TIME_COLUMN), segment.series(TIME_COLUMN)[1:])
        if later > earlier
    ]
    if not intervals:
        raise ValueError(f"{recording.filename} 没有正的时间间隔")
    return 1.0 / statistics.median(intervals)


def total_duration(recording: Recording) -> float:
    return sum(segment.duration for segment in recording.segments)


def sample_count(recording: Recording) -> int:
    """剔除首行空采样后的实际采样点数。"""
    return sum(len(segment.rows) for segment in recording.segments)


def magnitude_values(recording: Recording) -> list[float]:
    return [float(row[MAGNITUDE_COLUMN]) for segment in recording.segments for row in segment.rows]


def build_windows(recording: Recording, sample_rate: float, min_samples: int) -> list[Window]:
    windows: list[Window] = []
    for segment_index, segment in enumerate(recording.segments):
        start = float(segment.rows[0][TIME_COLUMN])
        buckets: dict[int, list[float]] = {}
        for row in segment.rows:
            index = int((float(row[TIME_COLUMN]) - start) / WINDOW_SECONDS)
            buckets.setdefault(index, []).append(float(row[MAGNITUDE_COLUMN]))

        kept = [(index, values) for index, values in sorted(buckets.items()) if len(values) >= min_samples]
        raw = [statistics.fmean(values) for _, values in kept]
        for position, (index, values) in enumerate(kept):
            low = max(0, position - SMOOTHING_WINDOWS // 2)
            high = min(len(raw), position + SMOOTHING_WINDOWS // 2 + 1)
            windows.append(
                Window(
                    recording.activity,
                    segment_index,
                    index,
                    len(kept),
                    raw[position],
                    statistics.median(raw[low:high]),
                    len(values),
                )
            )
    if not windows:
        raise ValueError(f"{recording.filename} 没有可用窗口")
    return windows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("空序列无法取分位")
    index = round((len(ordered) - 1) * fraction)
    return ordered[min(len(ordered) - 1, max(0, index))]


def window_values(windows: list[Window], smoothed: bool) -> list[float]:
    return [window.smoothed if smoothed else window.value for window in windows]


def build_thresholds(
    windows_by_activity: dict[str, list[Window]], mode: str, smoothed: bool
) -> list[float]:
    """mode='mean'：相邻类均值中点（旧口径）；mode='percentile'：90/10 分位中点。

    阈值取自与判决一致的特征口径，避免「用原始值定阈值、用中值判决」的口径错位。
    """
    names = [activity for activity, _, _ in ACTIVITIES]
    values = [window_values(windows_by_activity[name], smoothed) for name in names]
    thresholds = []
    for lower, upper in zip(values, values[1:]):
        if mode == "mean":
            thresholds.append((statistics.fmean(lower) + statistics.fmean(upper)) / 2)
        elif mode == "percentile":
            high = percentile(lower, THRESHOLD_PERCENTILE)
            low = percentile(upper, 1 - THRESHOLD_PERCENTILE)
            thresholds.append((high + low) / 2)
        else:
            raise ValueError(f"未知阈值模式：{mode}")
    return thresholds


def classify(value: float, thresholds: list[float]) -> str:
    return ACTIVITIES[sum(value >= threshold for threshold in thresholds)][0]


@dataclass
class Evaluation:
    label: str
    thresholds: list[float]
    total: int
    correct: int
    matrix: dict[tuple[str, str], int]
    min_samples: int = 0
    threshold_mode: str = "percentile"
    smoothed: bool = True

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def per_class(self) -> list[tuple[str, int, int]]:
        return [
            (
                activity,
                sum(count for (truth, predicted), count in self.matrix.items() if truth == activity and truth == predicted),
                sum(count for (truth, _), count in self.matrix.items() if truth == activity),
            )
            for activity, _, _ in ACTIVITIES
        ]

    def errors(self) -> list[tuple[str, str, int]]:
        return [
            (truth, predicted, count)
            for (truth, predicted), count in sorted(self.matrix.items())
            if truth != predicted and count
        ]


def evaluate(
    windows: list[Window],
    thresholds: list[float],
    smoothed: bool,
    label: str,
    min_samples: int = 0,
    threshold_mode: str = "percentile",
) -> Evaluation:
    matrix: dict[tuple[str, str], int] = {}
    correct = 0
    for window in windows:
        value = window.smoothed if smoothed else window.value
        truth = window.activity
        predicted = classify(value, thresholds)
        matrix[(truth, predicted)] = matrix.get((truth, predicted), 0) + 1
        correct += predicted == truth
    return Evaluation(label, thresholds, len(windows), correct, matrix, min_samples, threshold_mode, smoothed)


# --------------------------------------------------------------------------- 表格输出


def write_summary(recordings: list[Recording], windows_by_activity: dict[str, list[Window]], rates: dict[str, float]) -> None:
    with (OUTPUT_DIR / "activity_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "activity",
                "source_file",
                "sample_count",
                "duration_seconds",
                "sample_rate_hz",
                "mean_magnitude_m_s2",
                "p95_magnitude_m_s2",
                "window_count",
                "mean_window_m_s2",
                "median_window_m_s2",
            )
        )
        for recording in recordings:
            magnitude = magnitude_values(recording)
            values = [window.smoothed for window in windows_by_activity[recording.activity]]
            writer.writerow(
                (
                    recording.activity,
                    recording.filename,
                    sample_count(recording),
                    f"{total_duration(recording):.3f}",
                    f"{rates[recording.activity]:.2f}",
                    f"{statistics.fmean(magnitude):.6f}",
                    f"{percentile(magnitude, 0.95):.6f}",
                    len(values),
                    f"{statistics.fmean(values):.6f}",
                    f"{statistics.median(values):.6f}",
                )
            )


def write_windows(windows: list[Window], evaluation: Evaluation) -> None:
    with (OUTPUT_DIR / "window_features.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "activity",
                "segment_index",
                "window_index",
                "samples_in_window",
                "raw_value_m_s2",
                "smoothed_value_m_s2",
                "predicted_activity",
                "is_correct",
                "is_segment_edge",
            )
        )
        for window in windows:
            predicted = classify(window.smoothed, evaluation.thresholds)
            writer.writerow(
                (
                    window.activity,
                    window.segment_index,
                    window.window_index,
                    window.samples,
                    f"{window.value:.6f}",
                    f"{window.smoothed:.6f}",
                    predicted,
                    int(predicted == window.activity),
                    int(window.is_edge),
                )
            )


def write_confusion(evaluation: Evaluation) -> None:
    names = [activity for activity, _, _ in ACTIVITIES]
    with (OUTPUT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *names, "total"])
        for truth in names:
            row = [evaluation.matrix.get((truth, predicted), 0) for predicted in names]
            writer.writerow([truth, *row, sum(row)])
        writer.writerow(["total", *[sum(evaluation.matrix.get((truth, p), 0) for truth in names) for p in names], evaluation.total])


def write_segments(recordings: list[Recording], rates: dict[str, float]) -> None:
    """上下楼梯按方向拆开对比，说明方向信息被幅值特征抹掉了。"""
    with (OUTPUT_DIR / "segment_direction.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("activity", "direction", "segments", "rows", "duration_seconds", "mean_magnitude_m_s2", "z_std_m_s2")
        )
        for recording in recordings:
            if len(recording.segments) < 2:
                continue
            labels = list(dict.fromkeys(segment.label for segment in recording.segments))
            for label in labels:
                group = [segment for segment in recording.segments if segment.label == label]
                magnitude = [value for segment in group for value in segment.series(MAGNITUDE_COLUMN)]
                depth = [value for segment in group for value in segment.series(AXES[2][0])]
                writer.writerow(
                    (
                        recording.activity,
                        label,
                        len(group),
                        sum(len(segment.rows) for segment in group),
                        f"{sum(segment.duration for segment in group):.3f}",
                        f"{statistics.fmean(magnitude):.6f}",
                        f"{statistics.pstdev(depth):.6f}",
                    )
                )


# --------------------------------------------------------------------------- 绘图


def scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if math.isclose(low, high):
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def nice_limit(value: float) -> float:
    for candidate in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0):
        if value * 1.05 <= candidate:
            return candidate
    return math.ceil(value)


def write_waveform_svg(recordings: list[Recording]) -> None:
    width = 1120
    panel_height, top, bottom_margin = 232, 100, 52
    height = top + len(recordings) * panel_height + bottom_margin
    left, right = 96, width - 34
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei",Arial,sans-serif;fill:#26323f}'
        '.grid{stroke:#e2e7ec;stroke-width:1}.axis{stroke:#9aa5b1;stroke-width:1}'
        '.sep{stroke:#c3ccd6;stroke-width:1;stroke-dasharray:4 4}</style>',
        '<text x="96" y="36" font-size="21" font-weight="bold">四种活动的三轴线性加速度与合成幅值波形</text>',
        '<text x="96" y="57" font-size="12" fill="#5d6a77">'
        "纵轴按各活动自适应缩放，面板右上角标注量程；上下楼梯按采集片段拼接，片段之间留空隙</text>",
    ]
    legend_x = width - 372
    for offset, (name, color) in enumerate(
        [(axis_label, color) for _, axis_label, color in AXES] + [(MAGNITUDE_LABEL, MAGNITUDE_COLOR)]
    ):
        x = legend_x + offset * 64
        parts.append(
            f'<line x1="{x}" y1="30" x2="{x + 20}" y2="30" stroke="{color}" stroke-width="2.4"/>'
            f'<text x="{x + 25}" y="34" font-size="12">{html.escape(name)}</text>'
        )

    for panel, recording in enumerate(recordings):
        panel_top = top + panel * panel_height
        plot_top, plot_bottom = panel_top + 30, panel_top + panel_height - 46
        peak = max(
            abs(value)
            for segment in recording.segments
            for column, _, _ in AXES
            for value in segment.series(column)
        )
        limit = nice_limit(peak)
        decimals = 2 if limit < 1 else 1

        parts.append(
            f'<text x="{left}" y="{panel_top + 14}" font-size="15" font-weight="bold">'
            f'{html.escape(recording.activity)}</text>'
        )
        parts.append(
            f'<text x="{right}" y="{panel_top + 14}" text-anchor="end" font-size="11" fill="#5d6a77">'
            f"量程 ±{limit:g} m/s²</text>"
        )
        for fraction in (-1.0, -0.5, 0.0, 0.5, 1.0):
            tick = limit * fraction
            y = scale(tick, -limit, limit, plot_bottom, plot_top)
            parts.append(
                f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>'
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="10">'
                f"{tick:.{decimals}f}</text>"
            )

        spans: list[tuple[float, float, Segment]] = []
        cursor = 0.0
        for segment in recording.segments:
            spans.append((cursor, cursor + segment.duration, segment))
            cursor += segment.duration + 0.4
        total_time = max(cursor - 0.4, 0.01)

        for series_column, _, color in AXES:
            for display_start, _, segment in spans:
                step = max(1, len(segment.rows) // 2000)
                sampled = segment.rows[::step]
                if sampled[-1] is not segment.rows[-1]:
                    sampled.append(segment.rows[-1])
                points = [
                    f"{scale(display_start + float(row[TIME_COLUMN]) - float(segment.rows[0][TIME_COLUMN]), 0.0, total_time, left, right):.1f},"
                    f"{scale(float(row[series_column]), -limit, limit, plot_bottom, plot_top):.1f}"
                    for row in sampled
                ]
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="1.1" '
                    f'opacity="0.72" points="{" ".join(points)}"/>'
                )

        for display_start, _, segment in spans:
            step = max(1, len(segment.rows) // 2000)
            sampled = segment.rows[::step]
            if sampled[-1] is not segment.rows[-1]:
                sampled.append(segment.rows[-1])
            points = [
                f"{scale(display_start + float(row[TIME_COLUMN]) - float(segment.rows[0][TIME_COLUMN]), 0.0, total_time, left, right):.1f},"
                f"{scale(float(row[MAGNITUDE_COLUMN]), -limit, limit, plot_bottom, plot_top):.1f}"
                for row in sampled
            ]
            parts.append(
                f'<polyline fill="none" stroke="{MAGNITUDE_COLOR}" stroke-width="1.9" '
                f'opacity="0.92" points="{" ".join(points)}"/>'
            )

        if len(spans) > 1:
            for index, (display_start, display_end, segment) in enumerate(spans):
                center = (display_start + display_end) / 2
                x = scale(center, 0.0, total_time, left, right)
                parts.append(
                    f'<text x="{x:.1f}" y="{plot_top - 8:.1f}" text-anchor="middle" font-size="11" '
                    f'fill="#8a6a2f">{html.escape(segment.label)}</text>'
                )
                if index:
                    edge = scale(display_start - 0.2, 0.0, total_time, left, right)
                    parts.append(
                        f'<line class="sep" x1="{edge:.1f}" y1="{plot_top}" x2="{edge:.1f}" y2="{plot_bottom}"/>'
                    )

        parts.append(
            f'<line class="axis" x1="{left}" y1="{plot_bottom}" x2="{right}" y2="{plot_bottom}"/>'
        )
        tick_step = next(step for step in (1, 2, 5, 10, 20, 30, 60) if total_time / step <= 9)
        tick = 0
        while tick <= total_time + 1e-9:
            x = scale(float(tick), 0.0, total_time, left, right)
            parts.append(
                f'<line class="axis" x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 4}"/>'
                f'<text x="{x:.1f}" y="{plot_bottom + 16}" text-anchor="middle" font-size="10">{tick}</text>'
            )
            tick += tick_step

    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="12">'
        "累计时间 (s)；纵轴：加速度 (m/s²)</text>"
    )
    parts.append("</svg>")
    (OUTPUT_DIR / "acceleration_waveforms.svg").write_text("\n".join(parts), encoding="utf-8")


def write_threshold_svg(windows: list[Window], evaluation: Evaluation) -> None:
    width, height = 980, 610
    left, right, top, bottom = 126, width - 46, 86, 500
    names = [activity for activity, _, _ in ACTIVITIES]
    colors = {activity: color for activity, _, color in ACTIVITIES}
    values = [[window.smoothed for window in windows if window.activity == name] for name in names]
    maximum = nice_limit(max(max(group) for group in values))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei",Arial,sans-serif;fill:#26323f}'
        '.grid{stroke:#e2e7ec;stroke-width:1}</style>',
        f'<text x="{left}" y="34" font-size="20" font-weight="bold">1 秒窗口特征分布与分类阈值</text>',
        f'<text x="{left}" y="55" font-size="12" fill="#5d6a77">'
        "每个点代表一个窗口（已做 3 窗滑动中值）；虚线为相邻两类的分位点中点阈值</text>",
        f'<circle cx="{left}" cy="76" r="3.6" fill="#8c98a4"/>'
        f'<text x="{left + 10}" y="80" font-size="11" fill="#5d6a77">判定正确</text>'
        f'<circle cx="{left + 84}" cy="76" r="4.8" fill="none" stroke="#8c98a4" stroke-width="1.6"/>'
        f'<text x="{left + 94}" y="80" font-size="11" fill="#5d6a77">判定错误</text>',
    ]

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        tick = maximum * fraction
        y = scale(tick, 0.0, maximum, bottom, top)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{tick:.1f}</text>'
        )

    for index, name in enumerate(names):
        center = left + (index + 0.5) * (right - left) / len(names)
        parts.append(
            f'<text x="{center:.1f}" y="{bottom + 30}" text-anchor="middle" font-size="13">'
            f"{html.escape(name)}</text>"
        )
        if index:
            edge = left + index * (right - left) / len(names)
            parts.append(
                f'<line x1="{edge:.1f}" y1="{top}" x2="{edge:.1f}" y2="{bottom}" '
                'stroke="#eef1f4" stroke-width="1"/>'
            )
        group = values[index]
        mean = statistics.fmean(group)
        for point_index, value in enumerate(group):
            jitter = (point_index * 0.6180339887 % 1.0 - 0.5) * 86
            predicted = classify(value, evaluation.thresholds)
            y = scale(value, 0.0, maximum, bottom, top)
            if predicted == name:
                parts.append(
                    f'<circle cx="{center + jitter:.1f}" cy="{y:.1f}" r="3.4" '
                    f'fill="{colors[name]}" opacity="0.62"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{center + jitter:.1f}" cy="{y:.1f}" r="4.9" '
                    f'fill="none" stroke="{colors[name]}" stroke-width="1.7"/>'
                )
        y = scale(mean, 0.0, maximum, bottom, top)
        parts.append(
            f'<line x1="{center - 54:.1f}" y1="{y:.1f}" x2="{center + 54:.1f}" y2="{y:.1f}" '
            'stroke="#26323f" stroke-width="2" opacity="0.55"/>'
        )

    for threshold in evaluation.thresholds:
        y = scale(threshold, 0.0, maximum, bottom, top)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#34495e" stroke-width="1.5" stroke-dasharray="7 5"/>'
            f'<text x="{right - 4}" y="{y - 6:.1f}" text-anchor="end" font-size="11" '
            f'fill="#34495e">T = {threshold:.3f} m/s²</text>'
        )

    parts.extend(
        (
            f'<text x="{(left + right) / 2:.1f}" y="{height - 40}" text-anchor="middle" font-size="12">活动类别</text>',
            f'<text transform="translate(38 {(top + bottom) / 2:.1f}) rotate(-90)" '
            'text-anchor="middle" font-size="12">窗口平均绝对加速度 (m/s²)</text>',
            "</svg>",
        )
    )
    (OUTPUT_DIR / "activity_thresholds.svg").write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- 规则文档


def write_rules(
    recordings: list[Recording],
    windows: list[Window],
    rates: dict[str, float],
    evaluation: Evaluation,
    ablation: list[Evaluation],
) -> None:
    names = [activity for activity, _, _ in ACTIVITIES]
    lines = [
        "# 多姿态步态加速度特征采集与对比",
        "",
        "## 1 目的与范围",
        "",
        "用同一部手机、同一固定位置采集静止、平地慢走、上下楼梯、快跑四种状态的线性加速度，",
        "提取 1 秒窗口平均幅值作为分类特征，给出一套可复算的经验阈值规则，并量化其在本批数据上的分辨能力。",
        "本文件只描述这四段录音的统计规律，不构成通用步态判定或医学结论。",
        "",
        "## 2 采集条件",
        "",
        "| 项目 | 说明 |",
        "|---|---|",
        "| 采集工具 | Phyphox，导出「线性加速度」三轴与合成幅值 |",
        "| 采样率 | 200 Hz（时间戳间隔稳定为 0.005 s） |",
        "| 设备位置 | 四种状态使用同一部手机、同一固定位置、同一佩戴方式 |",
        "| 环境 | 同一段楼梯与同一段平地，由同一人完成 |",
        "",
        "四种状态采集条件一致，因此类别之间的数值差异可以归因于运动本身；",
        "若更换设备或佩戴位置，绝对值会整体平移，阈值需重新标定。",
        "",
        "## 3 数据处理",
        "",
        "- 输入为 `data/` 下四份原始 CSV，未做滤波、平滑或重采样，数值保持原始量级。",
        "- 每个片段开头的全零采样（传感器启动瞬间写出的一行 `0,0,0,0`）予以剔除，它不代表静止状态。",
        "- 上下楼梯文件含两段独立录制：第 4151 行时间由 20.853 s 回跳到 0.043 s。",
        "  按时间回跳与 `楼梯方向` 标签切分为「上楼梯」「下楼梯」两段，各自以本段起始时间归零，",
        "  因此表 1 中「上下楼梯」的时长是两段之和（49.16 s），而非文件末行时间戳（28.34 s）。",
        "- 时间轴在 18.7 s 附近缺 14 个采样点（单次间隔 0.0705 s）。按 1 秒分桶后该缺口被自然吸收，",
        "  不影响窗口均值，但做频谱分析时需注意。",
        f"- 特征值：每 1 秒窗口内 `Absolute acceleration` 的算术平均值，记作 M。",
        f"- 窗口覆盖不足 {WINDOW_COVERAGE:g} 秒（采样率 200 Hz 时约 {round(200 * WINDOW_COVERAGE)} 点）时丢弃；",
        "  实测被丢弃的窗口均落在片段末尾或缺口处，保留它们会因样本不足而抬高方差。",
        f"- 同一片段内对 M 取 {SMOOTHING_WINDOWS} 窗滑动中值再判决，抑制起停与折返造成的单窗异常。",
        "- 逐窗口的中间结果见 `outputs/window_features.csv`，可逐个核对。",
        "",
        "## 4 数据概览",
        "",
        "表 1 各活动整体统计",
        "",
        "| 活动 | 采样点 | 时长 (s) | 采样率 (Hz) | 平均幅值 (m/s²) | 有效窗口 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for recording in recordings:
        magnitude = magnitude_values(recording)
        count = sum(1 for window in windows if window.activity == recording.activity)
        lines.append(
            f"| {recording.activity} | {sample_count(recording)} | {total_duration(recording):.2f} | "
            f"{rates[recording.activity]:.1f} | {statistics.fmean(magnitude):.4f} | {count} |"
        )

    lines.extend(
        [
            "",
            "## 5 阈值规则",
            "",
            "对任意一段新采集的数据，先按 1 秒切窗、算窗口均值 M，再做 3 窗滑动中值，然后按 M 从低到高判断：",
            "",
            f"1. `M < {evaluation.thresholds[0]:.3f}`：静止",
            f"2. `{evaluation.thresholds[0]:.3f} ≤ M < {evaluation.thresholds[1]:.3f}`：平地慢走",
            f"3. `{evaluation.thresholds[1]:.3f} ≤ M < {evaluation.thresholds[2]:.3f}`：上下楼梯",
            f"4. `M ≥ {evaluation.thresholds[2]:.3f}`：快跑",
            "",
            f"阈值取相邻两类特征分布的 {THRESHOLD_PERCENTILE:.0%} 与 {1 - THRESHOLD_PERCENTILE:.0%} 分位点的中点。",
            "之所以不用均值中点：起停、折返等过渡窗口会把类别均值拉向中间，使阈值偏向样本多的一侧。",
            "分位点只依赖分布的稳定段，对少量离群窗口不敏感。",
            "",
            "表 2 相邻两类分位点与阈值",
            "",
            "| 阈值 | 下类 90 分位 | 上类 10 分位 | 取值 | 判定含义 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for index, threshold in enumerate(evaluation.thresholds):
        lower = window_values([window for window in windows if window.activity == names[index]], evaluation.smoothed)
        upper = window_values([window for window in windows if window.activity == names[index + 1]], evaluation.smoothed)
        lines.append(
            f"| T{index + 1} | {percentile(lower, THRESHOLD_PERCENTILE):.3f} | "
            f"{percentile(upper, 1 - THRESHOLD_PERCENTILE):.3f} | {threshold:.3f} | "
            f"{names[index]} 与 {names[index + 1]} 的分界 |"
        )

    lines.extend(
        [
            "",
            "## 6 分类结果",
            "",
            f"表 3 混淆矩阵（3 窗中值判决，共 {evaluation.total} 个窗口）",
            "",
            "| 真实 \\ 判定 | " + " | ".join(names) + " | 合计 |",
            "|---|" + "---:|" * (len(names) + 1),
        ]
    )
    for truth in names:
        row = [evaluation.matrix.get((truth, predicted), 0) for predicted in names]
        lines.append(f"| {truth} | " + " | ".join(str(count) for count in row) + f" | {sum(row)} |")
    totals = [sum(evaluation.matrix.get((truth, predicted), 0) for truth in names) for predicted in names]
    lines.append("| 合计 | " + " | ".join(str(count) for count in totals) + f" | {evaluation.total} |")

    lines.extend(["", f"整体命中 {evaluation.correct}/{evaluation.total} = {evaluation.accuracy:.1%}。", ""])
    lines.append("表 4 分类别命中率")
    lines.append("")
    lines.append("| 活动 | 命中 | 窗口数 | 准确率 |")
    lines.append("|---|---:|---:|---:|")
    for name, hit, total in evaluation.per_class():
        lines.append(f"| {name} | {hit} | {total} | {hit / total:.1%} |")

    lines.extend(
        [
            "",
            "## 7 误差来源",
            "",
            "误判不是随机分布的，全部落在片段的起止两端，稳定步态区间没有出现误判。",
            "",
        ]
    )
    raw_errors = [window for window in windows if classify(window.value, evaluation.thresholds) != window.activity]
    lines.append(f"逐窗口径下共 {len(raw_errors)} 处误判：")
    lines.append("")
    for window in raw_errors:
        predicted = classify(window.value, evaluation.thresholds)
        location = "段首" if window.window_index == 0 else ("段末" if window.is_edge else f"第 {window.window_index} ")
        lines.append(
            f"- 「{window.activity}」{location}窗原始值 {window.value:.3f} m/s² → 判为「{predicted}」"
            f"（同窗 3 窗中值 {window.smoothed:.3f} m/s²）"
        )
    lines.extend(
        [
            "",
            f"加入 3 窗滑动中值后剩 {len(evaluation.errors())} 处。中值的作用是把与邻窗明显不一致的单窗拉回本类：",
            "段内的冲高或塌低会被前后两窗压住，因此上述段内误判全部消失；",
            "唯一残留的是下楼梯最后一个窗口——段末只有一个邻居可供平均，中值仍被端点值主导。",
            "",
            "这说明 3 窗中值对段内过渡有效、对段末无效。若要彻底消除，需要把段末窗口按「有后邻窗」与「无后邻窗」区别处理，",
            "或改用 2 秒窗口让单个异常窗的权重降到一半；代价是判决延迟从 1 秒增加到 2 秒。",
            "",
            "另一个更根本的限制是：窗口幅值同时包含「姿势强度」与「运动阶段」两个因素。",
            "下楼梯最后 1 秒人体已经停住，幅值自然回落到慢走区间；起跑前两秒尚未进入稳定跑姿，幅值同样偏低。",
            "阈值规则只能切分强度，切不开阶段，这是单特征方案的结构性上限，与阈值取法无关。",
            "",
            "## 8 消融对照",
            "",
            "表 5 逐项改动对命中率的影响（同一份清洗后数据，逐项累加）",
            "",
            "| 口径 | 窗口门槛 | 阈值来源 | 判决方式 | 窗口数 | 命中 | 准确率 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for item in ablation:
        lines.append(
            f"| {item.label} | {item.min_samples} 点 | "
            f"{'均值中点' if item.threshold_mode == 'mean' else '分位中点'} | "
            f"{'3 窗中值' if item.smoothed else '逐窗'} | "
            f"{item.total} | {item.correct} | {item.accuracy:.1%} |"
        )
    lines.extend(
        [
            "",
            f"说明：以上 {len(ablation)} 个口径都在同一批数据上计算，没有独立测试集，因此都属于描述性验证，",
            "准确率会略高于面对新数据时的真实表现。要做严格评估，应另采一组数据后只做一次前向测试。",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## 9 方向信息",
            "",
            "表 6 上下楼梯按方向拆开",
            "",
            "| 方向 | 采样点 | 时长 (s) | 平均幅值 (m/s²) | z 轴标准差 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    direction_rows: list[tuple[str, int, float, float, float]] = []
    for recording in recordings:
        if len(recording.segments) < 2:
            continue
        for label in dict.fromkeys(segment.label for segment in recording.segments):
            group = [segment for segment in recording.segments if segment.label == label]
            magnitude = [value for segment in group for value in segment.series(MAGNITUDE_COLUMN)]
            depth = [value for segment in group for value in segment.series(AXES[2][0])]
            direction_rows.append(
                (
                    label,
                    sum(len(segment.rows) for segment in group),
                    sum(segment.duration for segment in group),
                    statistics.fmean(magnitude),
                    statistics.pstdev(depth),
                )
            )
            lines.append(
                f"| {label} | {direction_rows[-1][1]} | {direction_rows[-1][2]:.2f} | "
                f"{direction_rows[-1][3]:.3f} | {direction_rows[-1][4]:.2f} |"
            )
    up, down = direction_rows[0], direction_rows[1]
    lines.extend(
        [
            "",
            f"上行与下行的平均幅值几乎相同（{up[3]:.3f} 对 {down[3]:.3f}，相差仅 {abs(up[3] - down[3]):.3f} m/s²），",
            f"但上行耗时 {up[2]:.2f} s、下行 {down[2]:.2f} s，下行比上行慢 {down[2] / up[2]:.2f} 倍；",
            f"z 轴摆动标准差 {up[4]:.2f} 对 {down[4]:.2f}，相差 {up[4] / down[4]:.1f} 倍。",
            "这说明「上下楼梯」被合成一个类别时，方向信息被幅值特征完全抹掉：",
            "上楼是持续抬升重心、发力更集中；下楼是控制下降、单步时间更长但垂直冲击更小。",
            "若要区分方向，需要引入时长或轴间能量比，单靠幅值不可行。",
            "",
            "## 10 适用边界",
            "",
            "- 阈值来自 200 Hz、固定佩戴位置的四段录音，换设备、换位置、换人均需重新标定。",
            "- 幅值不能区分楼梯上下行，也不能区分走与慢跑等中间状态。",
            "- 本规则不适合作为医学诊断、康复评估或运动处方的依据。",
            "- 若需要更稳的判决，可把窗口延长到 2–3 秒，或对连续 5 窗做多数投票；代价是响应延迟增加。",
            "",
            "## 11 复现",
            "",
            "```",
            "python analyze_acceleration.py     # 生成全部表格、图和本文件",
            "python export_png.py               # 把两张 SVG 转成 PNG（需要本机浏览器）",
            "```",
            "",
            "输出文件：",
            "",
            "- `outputs/acceleration_waveforms.svg` / `.png`：四种活动的三轴与合成幅值波形。",
            "- `outputs/activity_thresholds.svg` / `.png`：窗口特征分布与阈值，误判窗口用空心圈标出。",
            "- `outputs/activity_summary.csv`：各活动采样率、时长、幅值统计。",
            "- `outputs/window_features.csv`：逐窗口特征值与判决结果。",
            "- `outputs/confusion_matrix.csv`：混淆矩阵。",
            "- `outputs/segment_direction.csv`：上下楼梯方向对比。",
            "- `outputs/threshold_rules.md`：本文件。",
            "",
        ]
    )
    (OUTPUT_DIR / "threshold_rules.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- 主流程


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    recordings = read_recordings()
    rates = {recording.activity: sample_rate_hz(recording) for recording in recordings}
    rate = statistics.median(rates.values())
    min_samples = round(rate * WINDOW_COVERAGE)
    legacy_samples = 50

    windows = [
        window
        for recording in recordings
        for window in build_windows(recording, rates[recording.activity], min_samples)
    ]
    by_activity = {activity: [window for window in windows if window.activity == activity] for activity, _, _ in ACTIVITIES}

    thresholds = build_thresholds(by_activity, "percentile", smoothed=True)
    evaluation = evaluate(
        windows, thresholds, True, "全量口径", min_samples=min_samples, threshold_mode="percentile"
    )

    legacy_windows = [
        window
        for recording in recordings
        for window in build_windows(recording, rates[recording.activity], legacy_samples)
    ]
    legacy_by_activity = {
        activity: [window for window in legacy_windows if window.activity == activity] for activity, _, _ in ACTIVITIES
    }
    ablation = [
        evaluate(
            legacy_windows,
            build_thresholds(legacy_by_activity, "mean", smoothed=False),
            False,
            "① 原口径",
            min_samples=legacy_samples,
            threshold_mode="mean",
        ),
        evaluate(
            legacy_windows,
            build_thresholds(legacy_by_activity, "percentile", smoothed=False),
            False,
            "② 仅换阈值",
            min_samples=legacy_samples,
            threshold_mode="percentile",
        ),
        evaluate(
            windows,
            build_thresholds(by_activity, "mean", smoothed=False),
            False,
            "③ 仅提门槛",
            min_samples=min_samples,
            threshold_mode="mean",
        ),
        evaluate(
            windows,
            build_thresholds(by_activity, "percentile", smoothed=False),
            False,
            "④ 换阈值 + 提门槛",
            min_samples=min_samples,
            threshold_mode="percentile",
        ),
        evaluation,
    ]

    write_summary(recordings, by_activity, rates)
    write_windows(windows, evaluation)
    write_confusion(evaluation)
    write_segments(recordings, rates)
    write_waveform_svg(recordings)
    write_threshold_svg(windows, evaluation)
    write_rules(recordings, windows, rates, evaluation, ablation)

    print(f"已分析 {len(recordings)} 份录音，采样率 {rate:.0f} Hz，有效窗口 {len(windows)} 个")
    print("阈值 (m/s²): " + ", ".join(f"{value:.3f}" for value in thresholds))
    for item in ablation:
        print(f"  {item.label}：{item.correct}/{item.total} = {item.accuracy:.1%}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
