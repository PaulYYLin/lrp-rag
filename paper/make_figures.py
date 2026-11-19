"""Render the bilingual paper figures from the saved experiment reports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
BLUE = "#176387"
ORANGE = "#a94814"
GRAY = "#586477"
INK = "#1c2939"
GRID = "#e0e6ec"
METHODS = {
    "raw768": ("Original 768", "原始 768 維", GRAY),
    "pca128": ("PCA 128", "PCA 128 維", ORANGE),
    "pca256": ("PCA 256", "PCA 256 維", "#85468e"),
    "learned128": ("Learned 128", "學習式 128 維", BLUE),
    "learned256": ("Learned 256", "學習式 256 維", "#14765e"),
}


def hotpot_points(records):
    return [(rep, mode) for rep in METHODS for mode in ("flat", "hnsw-m32")
            if f"{rep}-{mode}" in records]


def method_name(rep, mode, zh):
    en, tc, _ = METHODS[rep]
    if zh:
        return f"{tc}　{'精確' if mode == 'flat' else 'HNSW'}"
    return f"{en} {'exact' if mode == 'flat' else 'HNSW'}"


def interval(ax, y, center, low, high, color, marker="o"):
    ax.errorbar(center, y, xerr=[[center - low], [high - center]],
                fmt=marker, color=color, capsize=3, markersize=6, zorder=3)


def finish(fig, out, name, zh):
    lang = "zh-TW" if zh else "en"
    fig.savefig(out / f"{name}.{lang}.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trec(analysis, points, out, zh):
    scales = sorted(analysis["sizes"], key=int)
    x = range(len(scales))
    fig, (quality, latency) = plt.subplots(1, 2, figsize=(9.6, 3.7), layout="constrained")
    for rep, marker in [("learned128", "o"), ("pca128", "s"), ("raw768", "^")]:
        values = [points[size, f"{rep}-flat-k10"] for size in scales]
        quality.plot(x, values, marker=marker, color=METHODS[rep][2], linewidth=1.8,
                     label=method_name(rep, "flat", zh))
    quality.set(ylim=(.28, .59), ylabel="nDCG@10", title="A. 排序品質" if zh else "A. Ranking quality")
    quality.set_xticks(list(x), [f"{int(s):,}" for s in scales], rotation=0)
    quality.set_xlabel("語料段落數" if zh else "Passages in corpus")
    quality.legend(loc="lower left", frameon=False)
    quality.grid(axis="y")

    latency.axvline(.9, color=ORANGE, linestyle="--", linewidth=1.4,
                    label="延遲降低 10%" if zh else "10% reduction threshold")
    latency.axvline(1, color=GRAY, linestyle=":", linewidth=1.2,
                    label="等速" if zh else "Equal latency")
    for y, size in enumerate(scales):
        ratio = analysis["sizes"][size]["comparisons"]["learned128_flat_vs_raw_graph"]["left_over_right_p95"]
        interval(latency, y, ratio["estimate"], ratio["lower"], ratio["upper"], BLUE)
    latency.set_yticks(list(x), [f"{int(s):,}" for s in scales])
    latency.set(xlim=(0, 1.12), ylim=(2.65, -.55), title="B. 單題延遲比值" if zh else "B. Single-query latency ratio",
                xlabel="學習式 128 維精確 ÷ 原始 768 維 HNSW" if zh else "Learned 128 exact ÷ Original 768 HNSW")
    latency.legend(loc="lower left", frameon=False)
    latency.grid(axis="x")
    finish(fig, out, "trec-ranking-latency", zh)


def plot_evidence(report, out, zh):
    points = hotpot_points(report["conditions"])
    fig, (rates, changes) = plt.subplots(1, 2, figsize=(9.6, .55 * len(points) + .55), layout="constrained",
                                         sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    margin = 100 * report["evidence_margin"]
    changes.axvline(margin, color=ORANGE, linestyle="--", linewidth=1.4,
                    label="保留門檻" if zh else "Retention margin")
    changes.axvline(0, color=GRAY, linewidth=1, linestyle=":")
    for row, (rep, mode) in enumerate(points):
        name = f"{rep}-{mode}"
        metric = report["conditions"][name]["all"]
        rate = 100 * metric["complete"]
        color = METHODS[rep][2]
        exact = mode == "flat"
        rates.barh(row, rate, color=color if exact else "white", edgecolor=color,
                   height=.53, linewidth=1.2, zorder=3)
        rates.text(rate + 1.2, row, f"{rate:.1f}%", va="center", color=INK)
        effect = report["comparisons"][name]["complete"]
        low, high = effect["ci95"]
        interval(changes, row, 100 * effect["difference"], 100 * low, 100 * high,
                 color, "o" if exact else "s")
    rates.set_yticks(range(len(points)), [method_name(rep, mode, zh) for rep, mode in points])
    rates.set(xlim=(0, 64), ylim=(len(points) - .28, -.65),
              xlabel="完整證據率（%）" if zh else "Complete evidence rate (%)",
              title="A. 找齊全部支持句的比例" if zh else "A. Questions with all supporting sentences")
    changes.set(xlim=(-47, 5), xlabel="相對基準差值（百分點）" if zh else "Difference from baseline (percentage points)",
                title="B. 差值與配對 95% 區間" if zh else "B. Difference and paired 95% interval")
    changes.legend(loc="lower left", frameon=False)
    for ax in (rates, changes):
        ax.grid(axis="x")
    finish(fig, out, "hotpot-evidence", zh)


def plot_latency(timing, out, zh):
    points = hotpot_points(timing["points"])
    fig, (encoded, live) = plt.subplots(1, 2, figsize=(9.6, .52 * len(points) + .55),
                                       layout="constrained", sharey=True)
    panels = [
        (encoded, "pipeline_p95_ms", "A. 從已編碼向量開始" if zh else "A. From an encoded query",
         [.25, .5, 1, 2, 4, 8], (.20, 13), 1.12, 3, "log"),
        (live, "text_to_ids_p95_ms", "B. 從查詢文字開始" if zh else "B. From query text",
         [0, 5, 10, 15, 20, 25], (0, 29), None, 2, "linear"),
    ]
    for ax, key, title, ticks, limits, offset, digits, scale in panels:
        baseline = timing["points"]["raw768-flat"]["median"][key]
        ax.axvline(baseline, color=GRAY, linestyle=":", linewidth=1.1)
        for row, (rep, mode) in enumerate(points):
            value = timing["points"][f"{rep}-{mode}"]["median"][key]
            color = METHODS[rep][2]
            ax.scatter(value, row, marker="o" if mode == "flat" else "s", color=color, zorder=3)
            ax.text(value * offset if offset else value + .55, row, f"{value:.{digits}f}", va="center", color=INK)
        latencies = [timing["points"][f"{rep}-{mode}"]["median"][key] for rep, mode in points]
        ax.set(xscale=scale, title=title, xlabel="單題 p95（毫秒）" if zh else "Single-query p95 (ms)")
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.set_xlim(min(limits[0], min(latencies) * .8), max(limits[1], max(latencies) * 1.2))
        ax.grid(axis="x")
    encoded.xaxis.set_minor_formatter(NullFormatter())
    encoded.set_yticks(range(len(points)), [method_name(rep, mode, zh) for rep, mode in points])
    encoded.set_ylim(len(points) - .45, -.55)
    finish(fig, out, "hotpot-latency", zh)


def pareto_front(points):
    """Retain increases in evidence after sorting by lower latency first."""
    front = []
    for point in sorted(points, key=lambda point: (point[0], -point[1])):
        if not front or point[1] > front[-1][1]:
            front.append(point)
    return front


def plot_evidence_latency(report, timing, out, zh):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout="constrained", sharey=True)
    panels = [
        ("pipeline_p95_ms", "A. 從已編碼向量開始" if zh else "A. From an encoded query"),
        ("text_to_ids_p95_ms", "B. 從查詢文字開始" if zh else "B. From query text"),
    ]
    methods = hotpot_points(timing["points"])
    for ax, (metric, title) in zip(axes, panels, strict=True):
        coordinates = []
        for rep, mode in methods:
            name = f"{rep}-{mode}"
            latency = timing["points"][name]["median"][metric]
            evidence = 100 * report["conditions"][name]["all"]["complete"]
            coordinates.append((latency, evidence, rep, mode))
        front = pareto_front(coordinates)
        ax.plot([x for x, _, _, _ in front], [y for _, y, _, _ in front], color=INK,
                linestyle="--", linewidth=1.2, zorder=2)
        for point in coordinates:
            latency, evidence, rep, mode = point
            ax.scatter(latency, evidence, color=METHODS[rep][2], marker="o" if mode == "flat" else "s",
                       s=75 if point in front else 50, edgecolor=INK if point in front else "white",
                       linewidth=.9, zorder=3)
            name = f"{rep}-{mode}"
            if (metric == "pipeline_p95_ms" and name != report["reference"]
                    and report["conditions"][name]["passes_evidence_margin"]):
                ax.annotate("通過證據門檻" if zh else "Meets evidence criterion", (latency, evidence),
                            xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=title, ylim=(7, 60), xlabel="單題 p95（毫秒）" if zh else "Single-query p95 (ms)")
        ax.grid(True)
    encoded, live = axes
    encoded.set_xscale("log")
    ticks = [.25, .5, 1, 2, 4, 8]
    encoded.set_xticks(ticks, [str(v) for v in ticks])
    warm_latencies = [point["median"]["pipeline_p95_ms"] for point in timing["points"].values()]
    encoded.set_xlim(min(.21, min(warm_latencies) * .8), max(10, max(warm_latencies) * 1.2))
    encoded.xaxis.set_minor_formatter(NullFormatter())
    encoded.set_xlabel("單題 p95（毫秒，對數）" if zh else "Single-query p95 (ms, log scale)")
    live_latencies = [point["median"]["text_to_ids_p95_ms"] for point in timing["points"].values()]
    live.set_xlim(min(live_latencies) - 1, max(live_latencies) + 1)
    encoded.set_ylabel("完整證據率（%）" if zh else "Complete evidence rate (%)")
    handles = [Line2D([], [], color=METHODS[rep][2], linewidth=3, label=METHODS[rep][int(zh)])
               for rep in dict.fromkeys(rep for rep, _ in methods)]
    handles.extend([
        Line2D([], [], color=INK, marker="o", linestyle="none", label="精確搜尋" if zh else "Exact search"),
        Line2D([], [], color=INK, marker="s", linestyle="none", label="HNSW"),
        Line2D([], [], color=INK, linestyle="--", label="速度與證據前緣" if zh else "Speed–evidence frontier"),
    ])
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    finish(fig, out, "hotpot-evidence-latency", zh)


def main():
    report_dir = ARTIFACTS / "phase4-report-pca256"
    timing_dir = ARTIFACTS / "phase4-timing-pca256"
    if not ((report_dir / "summary.json").is_file() and (timing_dir / "timing.json").is_file()):
        report_dir = ARTIFACTS / "phase4-report"
        timing_dir = ARTIFACTS / "phase4-timing"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trec-report", type=Path, default=ARTIFACTS / "phase3-report")
    parser.add_argument("--hotpot-report", type=Path, default=report_dir)
    parser.add_argument("--hotpot-timing", type=Path, default=timing_dir)
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    parser.add_argument("--zh-font", type=Path, help="Chinese font file, for example Noto Sans CJK TC")
    args = parser.parse_args()

    tc_font = args.zh_font
    if tc_font is None:
        try:
            tc_font = font_manager.findfont(
                font_manager.FontProperties(family=["Noto Sans CJK TC", "Heiti TC", "PingFang TC"]),
                fallback_to_default=False,
            )
        except ValueError:
            parser.error("Use --zh-font to specify a Traditional Chinese font file.")
    font_manager.fontManager.addfont(tc_font)
    tc_family = font_manager.FontProperties(fname=tc_font).get_name()

    trec = json.loads((args.trec_report / "analysis.json").read_text(encoding="utf-8"))
    quality = (args.hotpot_report / "summary.json").read_bytes()
    hotpot = json.loads(quality)
    timing = json.loads((args.hotpot_timing / "timing.json").read_text(encoding="utf-8"))
    if timing["quality_sha256"] != hashlib.sha256(quality).hexdigest():
        parser.error("HotpotQA quality and timing must come from the same set of retrieval runs.")
    with (args.trec_report / "operating-points.csv").open(encoding="utf-8", newline="") as source:
        points = {(row["passages"], row["setting"]): float(row["ndcg10"])
                  for row in csv.DictReader(source)
                  if row["quality_split"] == "trec_combined" and row["batch"] == "1" and row["k"] == "10"}

    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False,
                         "axes.edgecolor": GRID, "grid.color": GRID, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    for zh in (False, True):
        plt.rcParams["font.family"] = ["DejaVu Sans", tc_family] if zh else ["DejaVu Sans"]
        plot_trec(trec, points, args.out, zh)
        plot_evidence(hotpot, args.out, zh)
        plot_latency(timing, args.out, zh)
        plot_evidence_latency(hotpot, timing, args.out, zh)


if __name__ == "__main__":
    main()
