"""Summarize locked TREC comparisons, paired uncertainty and descriptive search curves."""
import csv
from itertools import product

import numpy as np

from .io import atomic_file, fingerprint, read_json, read_rows, run_directory, write_json

SPLITS = ("trec2019", "trec2020", "trec_combined")
METRICS = ("ndcg@10", "recall@10")
SPEED_LIMIT = 0.90


def interval(estimate, samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"estimate": float(estimate), "lower": float(lo), "upper": float(hi)}


class QueryPairs:
    def __init__(self, rows, seed):
        self.ids = [(r["split"], r["id"]) for r in rows]
        rng = np.random.default_rng(seed)
        self.groups = {}
        for split in SPLITS[:2]:
            positions = np.array([i for i, key in enumerate(self.ids) if key[0] == split])
            self.groups[split] = (positions, rng.choice(positions, (10000, len(positions))))
        self.groups["trec_combined"] = tuple(
            np.concatenate(arrays, axis=-1) for arrays in zip(*self.groups.values(), strict=True))

    def values(self, records):
        if self.ids != [(r["split"], r["id"]) for r in records]:
            raise ValueError("Mismatched query IDs for paired comparison")
        return {metric: np.array([r["metrics"][metric] for r in records])
                for metric in records[0]["metrics"]}

    def difference(self, left, right):
        result = {split: {} for split in self.groups}
        for metric in left.keys() & right.keys():
            diff = left[metric] - right[metric]
            for split, (rows, draw) in self.groups.items():
                result[split][metric] = interval(diff[rows].mean(), diff[draw].mean(axis=1))
        return result


def latency_ratio(left, right, metric="pipeline_p95_ms"):
    a, b = [np.array([r[metric] for r in point["rounds"]]) for point in (left, right)]
    if len(a) != len(b) or len(a) != 5:
        raise ValueError("Expected five paired timing rounds")
    draw = np.array(list(product(range(5), repeat=5)))
    return interval(np.median(a) / np.median(b), np.median(a[draw], axis=1) / np.median(b[draw], axis=1))


def advantage(left, right, gates):
    ratio = latency_ratio(left, right)
    eligible = all(gates[p["name"]]["passes"] for p in (left, right))
    if not eligible:
        winner = "quality_gate_failed"
    elif ratio["upper"] <= SPEED_LIMIT:
        winner = left["name"]
    elif 1 / ratio["lower"] <= SPEED_LIMIT:
        winner = right["name"]
    else:
        winner = "inconclusive"
    return dict(left=left["name"], right=right["name"], left_over_right_p95=ratio,
                right_minus_left_ms=right["median"]["pipeline_p95_ms"]-left["median"]["pipeline_p95_ms"],
                eligible=eligible, outcome=winner)


def pareto(points, metric):
    def xy(p):
        return p["median"]["pipeline_p95_ms"], p["batch_quality"]["1"]["trec_combined"][metric]
    frontier = []
    for p in points:
        x, y = xy(p)
        if not any((a <= x and b >= y and (a < x or b > y)) for a, b in map(xy, points)):
            frontier.append(p["name"])
    return sorted(frontier)


def table(path, rows):
    with atomic_file(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def curves(out, sizes, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = dict(raw768="#235789", pca128="#8856a7", learned128="#b24b1c")
    fig, axes = plt.subplots(len(sizes), 2, figsize=(10, 3.1*len(sizes)+0.8), squeeze=False)
    for row, (size, points) in enumerate(sizes.items()):
        for ax, metric in zip(axes[row], METRICS, strict=True):
            for rep in grid["representations"]:
                group = [p for p in points if p["representation"] == rep and p["k"] == 10]
                graph = sorted((p for p in group if p["m"] == grid["primary_m"]), key=lambda p:p["ef_search"])
                exact = next(p for p in group if p["m"] is None)
                xs = [p["median"]["pipeline_p95_ms"] for p in graph]
                ys = [p["batch_quality"]["1"]["trec_combined"][metric] for p in graph]
                ax.plot(xs, ys, ".-", color=style[rep], lw=1.2)
                ax.scatter(exact["median"]["pipeline_p95_ms"],
                           exact["batch_quality"]["1"]["trec_combined"][metric],
                           marker="s", facecolors="none", edgecolors=style[rep])
                for x, y, p in zip(xs, ys, graph, strict=True):
                    if p["locked"]:
                        ax.scatter(x, y, s=58, facecolors="none", edgecolors=style[rep])
            ax.set(xscale="log", title=f"{size:,} passages", ylabel=metric, xlabel="Single-query p95 (ms)")
            ax.grid(alpha=0.25)
    handles = [Line2D([], [], color=color, label=rep) for rep,color in style.items()]
    handles += [Line2D([], [], color="black", marker=m, fillstyle="none", lw=0, label=label)
                for m,label in (("s","FlatIP"), ("o","Dev-selected HNSW"))]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "quality-latency.png", dpi=180)
    plt.close(fig)


def report(args):
    lock = read_json(args.lock)
    sizes, analysis, sweep, index_rows, inputs = {}, {}, [], [], {}
    verification = {}
    with run_directory(args.out) as out:
        for size_key, settings in sorted(lock["sizes"].items(), key=lambda item:int(item[0])):
            size = int(size_key)
            directory = args.results / f"phase3-trec-{size}"
            measured = directory / "measurement"
            summary = read_json(measured / "summary.json")
            if summary["passages"] != size or summary["lock_sha256"] != fingerprint(args.lock):
                raise ValueError("Wrong measurement or operating point lock")
            if args.trec_eval:
                from .phase3_verify import verify
                verification[size] = verify(directory, args.prepared, summary, args.trec_eval)
            inputs[size] = fingerprint(measured / "summary.json")
            points = summary["configurations"]
            sizes[size] = points
            by_name = {p["name"]:p for p in points}
            reps = lock["formal_test_grid"]["representations"]
            locked = {p["name"] for k in ("k10", "k100") for p in settings[k].values()}
            matched = settings["rq3_latency_matched"]
            matched_names = {p["name"] for p in matched.values() if p}
            reference = "raw768-flat-k10"
            pairs = QueryPairs(list(read_rows(measured / f"{reference}-batch1.jsonl")), lock["index_seed"])
            scores = {}
            for p in points:
                name = p["name"]
                p["locked"] = not p["m"] or name in locked
                if p["locked"] or name in matched_names:
                    scores[name] = pairs.values(list(read_rows(measured / f"{name}-batch1.jsonl")))
                for batch in (1, 32):
                    for split, quality in p["batch_quality"][str(batch)].items():
                        med = p["median"]
                        sweep.append(dict(passages=size, setting=name, m=p["m"], ef=p["ef_search"],
                                          k=p["k"], locked=p["locked"], batch=batch, quality_split=split,
                                          ndcg10=quality.get("ndcg@10"), recall10=quality.get("recall@10"),
                                          recall100=quality.get("recall@100"),
                                          p50_ms=med["pipeline_p50_ms" if batch == 1 else "batch32_p50_ms"],
                                          p95_ms=med["pipeline_p95_ms" if batch == 1 else "batch32_p95_ms"],
                                          qps=med["single_queries_per_second" if batch == 1 else "batch32_queries_per_second"],
                                          search_p95_ms=med["search_p95_ms"] if batch == 1 else None,
                                          uncached_p95_ms=med.get("encoding_pipeline_p95_ms") if batch == 1 else None))
            for name, cost in summary["indexes"].items():
                index_rows.append(dict(passages=size, index=name, **cost))
            gates = {}
            for name in scores:
                baseline = reference if by_name[name]["k"] == 10 else "raw768-flat-k100"
                diff = pairs.difference(scores[name], scores[baseline])
                gates[name] = dict(vs_raw_flat=diff, passes=all(
                    d["lower"] >= -lock["quality_margin"] for d in diff["trec_combined"].values()))

            comparisons = {}
            raw_hnsw = settings["k10"]["raw768"]["name"]
            for rep in reps:
                flat, graph = f"{rep}-flat-k10", settings["k10"][rep]["name"]
                comparisons[f"{rep}_graph_vs_flat"] = advantage(by_name[graph], by_name[flat], gates)
                if rep != "raw768":
                    comparisons[f"{rep}_flat_vs_raw_graph"] = advantage(by_name[flat], by_name[raw_hnsw], gates)
                    comparisons[f"{rep}_graph_vs_raw_graph"] = advantage(by_name[graph], by_name[raw_hnsw], gates)
            comparisons["learned_graph_vs_pca_graph"] = advantage(
                by_name[settings["k10"]["learned128"]["name"]], by_name[settings["k10"]["pca128"]["name"]], gates)
            rq1 = {metric: pairs.difference(scores[f"learned128-flat-k{k}"], scores[f"pca128-flat-k{k}"])
                   for metric,k in (("k10",10),("k100",100))}
            speed_matched = {
                rep: pairs.difference(scores[item["name"]], scores[matched["raw768"]["name"]]) if item else None
                for rep, item in matched.items() if rep != "raw768"}
            graph_points = [p for p in points if p["k"] == 10 and p["m"] == lock["formal_test_grid"]["primary_m"]]
            analysis[size] = dict(
                judged=summary["judged"], background=summary["background"], gates=gates, comparisons=comparisons,
                rq1=rq1, rq3_latency_matched={r: p["name"] if p else None for r,p in matched.items()},
                rq3_latency_matched_difference=speed_matched,
                rq3_hnsw_frontiers={m: pareto(graph_points, m) for m in METRICS},
                rq3_graph_difference={rep: pairs.difference(scores[settings["k10"]["learned128"]["name"]],
                                                           scores[settings["k10"][rep]["name"]])
                                      for rep in ("raw768","pca128")},
                pareto={m: pareto([p for p in points if p["k"]==10 and p["m"] in (None,32)], m)
                        for m in METRICS})
        primary_host = read_json(args.results / f"phase3-trec-{lock['rq1_primary_passages']}"
                                 / "measurement/summary.json")["host"]
        full_count = read_json(args.prepared / "manifest.json")["collection_passages"]
        raw_bytes = full_count * 768 * np.dtype("float32").itemsize
        full = dict(passages=full_count, raw_vector_bytes=raw_bytes,
                    same_size_dev_lock=str(full_count) in lock["sizes"], host=primary_host,
                    fits_raw_vectors=raw_bytes <= primary_host["physical_memory_bytes"]
                    if "physical_memory_bytes" in primary_host else None)
        table(out / "operating-points.csv", sweep)
        table(out / "index-costs.csv", index_rows)
        write_json(out / "analysis.json", dict(lock_sha256=fingerprint(args.lock), measurements_sha256=inputs,
                                               primary_passages=lock["rq1_primary_passages"], full_corpus=full, sizes=analysis))
        if verification:
            write_json(out / "verification.json", {"scorer_sha256": fingerprint(args.trec_eval),
                                                  "sizes": verification})
        curves(out, sizes, lock["formal_test_grid"])
        render_report(out, sizes, analysis, lock, full)


def render_report(out, sizes, analysis, lock, full):
    labels = {"raw768": "原始 768 維", "pca128": "PCA 128 維", "learned128": "學習式 128 維"}
    def method(point):
        return labels[point["representation"]] + (" HNSW" if point["m"] else " 精確")

    lines = ["# 第三階段 TREC 檢索實測", "",
             "TREC 2019 的 43 題和 2020 的 54 題共用各規模的段落庫。合併成績以 97 題平均。"
             "每個規模保留兩年全部 20,349 個已判定段落，再按固定順序加入背景段落。"
             "三個規模的已判定段落分別占 67.83%、20.35%、6.783%。", "",
             "## 128 維精確檢索的排序品質", "",
             "下表是學習式投影減去 PCA 的差值。括號為以查詢配對、各年度內重抽 10,000 次的 95% 區間。"
             f"預先指定的主要比較使用 {lock['rq1_primary_passages']:,} 段。", "",
             "| 段落 | 測試集 | nDCG@10 差值 | Recall@10 差值 | Recall@100 差值 |",
             "| ---: | --- | --- | --- | --- |"]
    for size, result in analysis.items():
        for split in SPLITS:
            scores = {**result["rq1"]["k10"][split], **result["rq1"]["k100"][split]}
            cells = [f"{s['estimate']:+.4f} [{s['lower']:+.4f}, {s['upper']:+.4f}]"
                     for s in (scores[k] for k in (*METRICS, "recall@100"))]
            lines.append(f"| {size:,} | {split} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 各方法的品質、延遲及吞吐量", "",
                  "HNSW 採各規模在開發集鎖定的 M=32 操作點。品質欄使用單筆查詢結果。"
                  "p95 是單筆查詢從已編碼向量到傳回前十個段落 ID 的耗時。"
                  "含編碼 p95 另外從查詢文字開始，包含即時斷詞和模型推論。"
                  "批次吞吐量從已編碼的 32 筆查詢開始。各欄耗時和吞吐量均取五輪中位數。", "",
                  "| 段落 | 方法 | efSearch（k10 / k100） | nDCG@10 | Recall@10 | Recall@100 | p95 ms | 含編碼 p95 ms | 批次 32 QPS | 通過品質門檻 |",
                  "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for size, points in sizes.items():
        settings = lock["sizes"][str(size)]
        by_name = {p["name"]:p for p in points}
        for p in points:
            if not p["locked"] or p["k"] != 10:
                continue
            rep = p["representation"]
            p100 = by_name[settings["k100"][rep]["name"] if p["m"] else f"{rep}-flat-k100"]
            q, q100 = [x["batch_quality"]["1"]["trec_combined"] for x in (p,p100)]
            depth = f"{p['ef_search']} / {p100['ef_search']}" if p["m"] else "—"
            timing = p["median"]
            passed = "通過" if analysis[size]["gates"][p["name"]]["passes"] else "未達"
            lines.append(f"| {size:,} | {method(p)} | {depth} | {q['ndcg@10']:.4f} | {q['recall@10']:.4f} | "
                         f"{q100['recall@100']:.4f} | {timing['pipeline_p95_ms']:.3f} | "
                         f"{timing['encoding_pipeline_p95_ms']:.3f} | {timing['batch32_queries_per_second']:,.0f} | {passed} |")
    lines.extend(["", "## 學習式精確檢索與原始 HNSW 的速度優勢", "",
                  "兩方相對原始精確檢索的 nDCG@10、Recall@10 配對差值區間下界均須達 −0.01。"
                  "表中比值為學習式精確檢索 p95 除以原始 HNSW p95。"
                  "配對重抽五輪的計時結果，列舉全部 3,125 種樣本，計算兩個中位數的比值。"
                  "某一方的比值區間上界至多為 0.90，且兩方通過品質門檻，才判定該方有實質速度優勢。", "",
                  "| 段落 | 學習式精確 p95 ms | 原始 HNSW p95 ms | 比值 [95% 區間] | 原始 HNSW 減學習式精確 ms | 判定 |",
                  "| ---: | ---: | ---: | --- | ---: | --- |"])
    wins = []
    for size, result in analysis.items():
        c = result["comparisons"]["learned128_flat_vs_raw_graph"]
        by_name = {p["name"]:p for p in sizes[size]}
        a, b = [by_name[c[key]]["median"]["pipeline_p95_ms"] for key in ("left","right")]
        ratio = c["left_over_right_p95"]
        verdict = {c["left"]: "學習式精確較快", c["right"]: "原始 HNSW 較快",
                   "quality_gate_failed":"品質門檻未通過", "inconclusive":"速度差異未達判準"}[c["outcome"]]
        if c["outcome"] in (c["left"], c["right"]):
            wins.append((size, verdict))
        lines.append(f"| {size:,} | {a:.3f} | {b:.3f} | {ratio['estimate']:.3f} "
                     f"[{ratio['lower']:.3f}, {ratio['upper']:.3f}] | {c['right_minus_left_ms']:+.3f} | {verdict} |")
    flips = [(a[0], b[0]) for a,b in zip(wins,wins[1:]) if a[1]!=b[1]]
    lines.extend(["", "觀察到相反優勢的實測端點為 " + "、".join(f"{a:,} 與 {b:,} 段" for a,b in flips) + "。"
                  if flips else "這三個規模未觀察到有共同品質支持的速度優勢轉換。"])
    lines.extend(["", "## 索引成本與參數掃描", "",
                  "建置時間從建立空索引到加入全部向量。檔案大小取自序列化索引。RSS 增量是在新程序中單獨載入該索引前後的常駐記憶體差。"
                  "10 萬段另列 M=16、32、48 的建圖成本。"
                  "[各索引的秒數與位元組數](index-costs.csv)", "",
                  "下圖的方形為精確搜尋，折線連接 M=32 下由淺到深的 HNSW 操作點，圈選位置為開發集預選深度。"
                  "逐操作點表依年度與批次列出品質，耗時欄共用兩年 97 題的計時。表中也列出 10 萬段的 M 敏感性掃描。"
                  "[逐操作點表](operating-points.csv)", "",
                  "![各規模的 p95 延遲、nDCG@10 與 Recall@10](quality-latency.png)", "",
                  "同表示內 HNSW 與精確搜尋的比較、兩種低維 HNSW 的品質差值及兩個品質指標各自的 Pareto 前緣列於統計檔。"
                  "三個規模的開發集都未找到與原始 HNSW 延遲相差 5% 以內的低維 HNSW 操作點。"
                  "[配對區間與達標狀態](analysis.json)", ""])
    lines.extend(["## 學習式 HNSW 的速度與排名取捨", "",
                  "下表比較兩種在開發集選定的 HNSW。p95 比值為學習式 128 維除以原始 768 維，"
                  "品質差值則為學習式減去原始。最後一欄計算在 M=32、k=10 掃描中，"
                  "有幾個學習式 HNSW 深度位於 p95 與 nDCG@10 的前緣。", "",
                  "| 段落 | p95 比值 [95% 區間] | nDCG@10 差值 [95% 區間] | Recall@10 差值 [95% 區間] | 前緣點數 |",
                  "| ---: | --- | --- | --- | ---: |"])
    for size, result in analysis.items():
        ratio = result["comparisons"]["learned128_graph_vs_raw_graph"]["left_over_right_p95"]
        quality = result["rq3_graph_difference"]["raw768"]["trec_combined"]
        columns = [f"{item['estimate']:.4f} [{item['lower']:.4f}, {item['upper']:.4f}]"
                   for item in (ratio, quality["ndcg@10"], quality["recall@10"])]
        frontier = result["rq3_hnsw_frontiers"]["ndcg@10"]
        count = sum(name.startswith("learned128-") for name in frontier)
        lines.append(f"| {size:,} | " + " | ".join(columns) + f" | {count} |")
    lines.append("")
    if full["fits_raw_vectors"] is False:
        host = full["host"]
        lines.extend(["## 全庫複核的資源評估", "",
                      f"{full['passages']:,} 段的原始 768 維 float32 向量需 "
                      f"{full['raw_vector_bytes']/2**30:.2f} GiB。實測主機 {host['cpu']} 有 "
                      f"{host['physical_memory_bytes']/2**30:.0f} GiB 實體記憶體。"
                      "全庫的原始 HNSW 還需要圖連結和建圖工作空間。"
                      "本次未執行全庫同規模開發校準及 TREC 常駐索引複核，速度判定限於上列測點。", ""])
    with atomic_file(out / "results.md") as stream:
        stream.write("\n".join(lines))
