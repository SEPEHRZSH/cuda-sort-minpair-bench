# Result Plotter (stats + prettier plots + ratio/std) + analysis TXT export
# What it does:
#  - Reads outputs/stats.txt (preferred) or outputs/log.txt (fallback)
#  - Writes timings.csv (best/avg/median/std/min/max)
#  - Writes nice plots (PNG + optional SVG)
#  - NEW: Writes analysis_report.txt containing all numbers + derived comparisons
# Usage:
#   python plot_results_stats_pretty.py --log outputs/log.txt --out outputs/plots
#   python plot_results_stats_pretty.py --log outputs/log.txt --out outputs/plots --svg
# Notes:
#  - If outputs/stats.txt exists next to --log, it will be used automatically.
#  - Speedup is defined as shared / distributed:
#       speedup > 1  => distributed is faster
#       speedup < 1  => shared is faster

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


@dataclass
class StatsRow:
    part: str          # "A" or "B"
    variant: str       # "shared" or "distributed"
    n: int
    repeats: int
    best: float
    avg: float
    median: float
    std: float
    minv: float
    maxv: float
    even: Optional[int] = None
    odd: Optional[int] = None
    a: Optional[int] = None
    b: Optional[int] = None
    diff: Optional[int] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_blocks(text: str):
    return re.split(r"(?m)^\s*===\s*", text)


def parse_stats_or_log(text: str) -> Dict[Tuple[str, str], StatsRow]:
    rows: Dict[Tuple[str, str], StatsRow] = {}
    for blk in _parse_blocks(text):
        blk = blk.strip()
        if not blk:
            continue

        m = re.match(r"Part\s+([AB])\s*\(([^)]+)\)\s*===\s*(.*)$", blk, re.DOTALL)
        if not m:
            continue

        part = m.group(1).strip()
        variant = m.group(2).strip().lower()
        body = m.group(3)

        n = None
        even = odd = None
        a = b = diff = None

        if part == "A":
            mn = re.search(
                r"(?m)^\s*N\s*=\s*(\d+)\s+even_count\s*=\s*(\d+)\s+odd_count\s*=\s*(\d+)\s*$",
                body
            )
            if mn:
                n = int(mn.group(1))
                even = int(mn.group(2))
                odd = int(mn.group(3))
        else:
            mn = re.search(
                r"(?m)^\s*N\s*=\s*(\d+)\s+pair\s*=\s*\(([-\d]+)\s*,\s*([-\d]+)\)\s+diff\s*=\s*(\d+)\s*$",
                body
            )
            if mn:
                n = int(mn.group(1))
                a = int(mn.group(2))
                b = int(mn.group(3))
                diff = int(mn.group(4))

        if n is None:
            continue

        def fval(key: str) -> Optional[float]:
            mm = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([0-9.]+)\s*$", body)
            return float(mm.group(1)) if mm else None

        best = fval("best_ms")
        if best is None:
            continue

        mr = re.search(r"(?m)^\s*repeats\s*=\s*(\d+)\s*$", body)
        repeats = int(mr.group(1)) if mr else 1

        avg = fval("avg_ms")
        median = fval("median_ms")
        std = fval("std_ms")
        minv = fval("min_ms")
        maxv = fval("max_ms")

        if avg is None: avg = best
        if median is None: median = best
        if std is None: std = 0.0
        if minv is None: minv = best
        if maxv is None: maxv = best

        rows[(part, variant)] = StatsRow(
            part=part, variant=variant, n=n, repeats=repeats,
            best=best, avg=avg, median=median, std=std, minv=minv, maxv=maxv,
            even=even, odd=odd, a=a, b=b, diff=diff
        )

    return rows


def write_timings_csv(out_dir: str, rows: Dict[Tuple[str, str], StatsRow]) -> str:
    path = os.path.join(out_dir, "timings.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part", "variant", "n", "repeats", "best_ms", "avg_ms", "median_ms", "std_ms", "min_ms", "max_ms"])
        for k in sorted(rows.keys()):
            r = rows[k]
            w.writerow([
                r.part, r.variant, r.n, r.repeats,
                f"{r.best:.6f}", f"{r.avg:.6f}", f"{r.median:.6f}",
                f"{r.std:.6f}", f"{r.minv:.6f}", f"{r.maxv:.6f}"
            ])
    return path


def _ms_formatter():
    def fmt(x, _pos):
        if x < 10:
            return f"{x:.3f}"
        return f"{x:.2f}"
    return FuncFormatter(fmt)


def _setup_matplotlib(dpi: int):
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": max(220, dpi),
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    })


def _format_subtitle(n: Optional[int], repeats: Optional[int]) -> str:
    if n is None and repeats is None:
        return ""
    if n is None:
        return f"repeats={repeats}"
    if repeats is None:
        return f"N={n:,}"
    return f"N={n:,}   |   repeats={repeats}"


def _apply_titles(fig, title: str, subtitle: str):
    fig.suptitle(title, y=0.985, fontsize=13)
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=10)


def _annotate_bars(ax, xs, ys, fmt="{:.3f}", y_pad_frac=0.02):
    ymax = max(ys) if ys else 1.0
    ytop = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], max(ytop, ymax * (1.0 + y_pad_frac) + 1e-9))

    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            xy=(x, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
        )


def bar_with_whiskers(title: str, subtitle: str, labels, means, mins, maxs, medians,
                      ylabel: str, out_png: str, also_svg: bool, rotate_xticks: bool) -> None:
    fig = plt.figure(figsize=(8.2, 4.8) if len(labels) > 3 else (7.2, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    xs = list(range(len(labels)))
    ax.bar(xs, means)

    yerr_low = [max(0.0, m - lo) for m, lo in zip(means, mins)]
    yerr_high = [max(0.0, hi - m) for m, hi in zip(means, maxs)]
    ax.errorbar(xs, means, yerr=[yerr_low, yerr_high], fmt="none", capsize=6)

    ax.scatter(xs, medians)

    ax.set_xticks(xs)
    if rotate_xticks:
        ax.set_xticklabels(labels, rotation=15, ha="right")
    else:
        ax.set_xticklabels(labels)

    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(_ms_formatter())

    top_ref = max([m + e for m, e in zip(means, yerr_high)] + ([max(maxs)] if maxs else [0.0]))
    ax.set_ylim(0.0, top_ref * 1.22 + 1e-9)

    _annotate_bars(ax, xs, means)

    _apply_titles(fig, title, subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    fig.savefig(out_png)
    if also_svg:
        fig.savefig(out_png[:-4] + ".svg")
    plt.close(fig)


def speedup_mean_median(shared: StatsRow, dist: StatsRow) -> Tuple[float, float]:
    mean_sp = shared.avg / dist.avg if dist.avg > 0 else float("nan")
    med_sp = shared.median / dist.median if dist.median > 0 else float("nan")
    return mean_sp, med_sp


def speedup_std(shared: StatsRow, dist: StatsRow) -> float:
    # sigma_R ~= R * sqrt( (sigma_S/mu_S)^2 + (sigma_D/mu_D)^2 )
    if shared.avg <= 0 or dist.avg <= 0:
        return float("nan")
    r = shared.avg / dist.avg
    rel_s = (shared.std / shared.avg) if shared.avg > 0 else 0.0
    rel_d = (dist.std / dist.avg) if dist.avg > 0 else 0.0
    return abs(r) * math.sqrt(rel_s * rel_s + rel_d * rel_d)


def speedup_plot(title: str, subtitle: str, labels, mean_sps, median_sps, std_sps,
                 out_png: str, also_svg: bool) -> None:
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(1, 1, 1)

    xs = list(range(len(labels)))
    ax.bar(xs, mean_sps)
    if std_sps is not None:
        ax.errorbar(xs, mean_sps, yerr=std_sps, fmt="none", capsize=6)

    ax.scatter(xs, median_sps)
    ax.axhline(1.0, linestyle="--", linewidth=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup (shared / distributed)")

    if std_sps is not None and len(std_sps) == len(mean_sps):
        top_ref = max([m + (e if e == e else 0.0) for m, e in zip(mean_sps, std_sps)] + [1.0])
    else:
        top_ref = max(mean_sps + [1.0]) if mean_sps else 1.0
    ax.set_ylim(0.0, top_ref * 1.22 + 1e-9)

    _annotate_bars(ax, xs, mean_sps, fmt="{:.3f}x", y_pad_frac=0.06)

    _apply_titles(fig, title, subtitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])

    fig.savefig(out_png)
    if also_svg:
        fig.savefig(out_png[:-4] + ".svg")
    plt.close(fig)


def _pct_faster(speedup_shared_over_dist: float) -> float:
    if not (speedup_shared_over_dist > 0):
        return float("nan")
    return (1.0 - 1.0 / speedup_shared_over_dist) * 100.0


def write_analysis_report(out_dir: str,
                          title_prefix: str,
                          source_path: str,
                          rows: Dict[Tuple[str, str], StatsRow],
                          plots_written) -> str:
    path = os.path.join(out_dir, "analysis_report.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_any = None
    rep_any = None
    for k in rows:
        n_any = rows[k].n
        rep_any = rows[k].repeats
        break

    def line(s=""):
        return s + "\n"

    def fmt_ms(x: float) -> str:
        return f"{x:.6f}"

    def fmt_opt(x):
        return "NA" if x is None else str(x)

    with open(path, "w", encoding="utf-8") as f:
        f.write(line(f"{title_prefix} - Auto Analysis Export"))
        f.write(line(f"Generated: {now}"))
        f.write(line(f"Source parsed: {source_path}"))
        if n_any is not None:
            f.write(line(f"N: {n_any}"))
        if rep_any is not None:
            f.write(line(f"repeats: {rep_any}"))
        f.write(line())

        f.write(line("1) Parsed Results (ms)"))
        f.write(line("part,variant,n,repeats,best,avg,median,std,min,max,extra"))
        for (part, variant) in sorted(rows.keys()):
            r = rows[(part, variant)]
            if part == "A":
                extra = f"even={fmt_opt(r.even)} odd={fmt_opt(r.odd)}"
            else:
                extra = f"pair=({fmt_opt(r.a)},{fmt_opt(r.b)}) diff={fmt_opt(r.diff)}"
            f.write(line(
                f"{r.part},{r.variant},{r.n},{r.repeats},"
                f"{fmt_ms(r.best)},{fmt_ms(r.avg)},{fmt_ms(r.median)},{fmt_ms(r.std)},{fmt_ms(r.minv)},{fmt_ms(r.maxv)},"
                f"{extra}"
            ))
        f.write(line())

        f.write(line("2) Speedup (shared / distributed)"))
        f.write(line("Interpretation: speedup > 1 => distributed is faster; speedup < 1 => shared is faster"))
        f.write(line("part,speedup_mean,speedup_std_est,speedup_median,distributed_faster_by_%(vs_shared_mean)"))
        for part in ["A", "B"]:
            if (part, "shared") in rows and (part, "distributed") in rows:
                shared = rows[(part, "shared")]
                dist = rows[(part, "distributed")]
                sp_mean, sp_median = speedup_mean_median(shared, dist)
                sp_std = speedup_std(shared, dist)
                imp = _pct_faster(sp_mean)
                f.write(line(
                    f"{part},{sp_mean:.6f},{sp_std:.6f},{sp_median:.6f},{imp:.3f}"
                ))
        f.write(line())

        f.write(line("3) Report-ready Notes"))
        for part in ["A", "B"]:
            if (part, "shared") in rows and (part, "distributed") in rows:
                shared = rows[(part, "shared")]
                dist = rows[(part, "distributed")]
                sp_mean, sp_median = speedup_mean_median(shared, dist)
                imp = _pct_faster(sp_mean)

                winner = "distributed" if sp_mean > 1.0 else ("shared" if sp_mean < 1.0 else "tie")
                f.write(line(f"- Part {part}: mean(shared)={shared.avg:.6f} ms, mean(dist)={dist.avg:.6f} ms."))
                f.write(line(f"  -> speedup_mean(shared/dist)={sp_mean:.3f}x; median ratio={sp_median:.3f}x; winner={winner}."))
                f.write(line(f"  -> distributed improvement vs shared (mean) ≈ {imp:.2f}% (positive means distributed faster)."))

                def cov(r: StatsRow) -> float:
                    return (r.std / r.avg) if r.avg > 0 else float("nan")

                f.write(line(f"  Stability (CV=std/mean): shared={cov(shared):.4f}, dist={cov(dist):.4f}."))
                f.write(line(
                    f"  Range (min..max ms): shared=[{shared.minv:.6f}, {shared.maxv:.6f}], dist=[{dist.minv:.6f}, {dist.maxv:.6f}]."
                ))

                if part == "A":
                    f.write(line(f"  Output structure: even_count={shared.even} / odd_count={shared.odd} (should match dist)."))
                else:
                    f.write(line(f"  Closest pair found (shared): ({shared.a}, {shared.b}), diff={shared.diff} (should match dist)."))
                f.write(line())

        f.write(line("4) Files Produced"))
        f.write(line(f"- timings.csv: {os.path.join(out_dir, 'timings.csv')}"))
        f.write(line(f"- analysis_report.txt: {path}"))
        for p in plots_written:
            f.write(line(f"- plot: {p}"))

    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join("outputs", "log.txt"),
                    help="Path to outputs/log.txt (stats.txt auto-detected next to it)")
    ap.add_argument("--out", default=os.path.join("outputs", "plots"),
                    help="Output folder for plots/csv/report")
    ap.add_argument("--svg", action="store_true",
                    help="Also write SVG copies of plots (good for reports)")
    ap.add_argument("--dpi", type=int, default=150,
                    help="Figure DPI for rendering (default: 150)")
    ap.add_argument("--title-prefix", default="HW3",
                    help="Prefix added to plot titles")
    args = ap.parse_args()

    _setup_matplotlib(args.dpi)

    log_path = args.log
    base_dir = os.path.dirname(os.path.abspath(log_path))
    stats_path = os.path.join(base_dir, "stats.txt")

    chosen = stats_path if os.path.exists(stats_path) else log_path
    if not os.path.exists(chosen):
        print(f"[ERROR] file not found: {chosen}")
        return 1

    ensure_dir(args.out)

    with open(chosen, "r", encoding="utf-8") as f:
        txt = f.read()

    rows = parse_stats_or_log(txt)
    if not rows:
        print("[ERROR] Could not parse results (format unexpected).")
        return 2

    csv_path = write_timings_csv(args.out, rows)
    print("[OK] wrote:", csv_path)

    n_any = None
    rep_any = None
    for k in rows:
        n_any = rows[k].n
        rep_any = rows[k].repeats
        break
    subtitle_common = _format_subtitle(n_any, rep_any)

    plots_written = []

    for part in ["A", "B"]:
        variants = [v for v in ["distributed", "shared"] if (part, v) in rows]
        if not variants:
            continue

        labels = variants
        means = [rows[(part, v)].avg for v in variants]
        mins = [rows[(part, v)].minv for v in variants]
        maxs = [rows[(part, v)].maxv for v in variants]
        meds = [rows[(part, v)].median for v in variants]

        out_png = os.path.join(args.out, f"part{part}_time_stats.png")
        bar_with_whiskers(
            f"{args.title_prefix} Part {part} runtime (bar=mean, whisker=min/max, dot=median)",
            subtitle_common, labels, means, mins, maxs, meds,
            "Time (ms)", out_png, args.svg, rotate_xticks=False
        )
        print("[OK] wrote:", out_png)
        plots_written.append(out_png)
        if args.svg:
            plots_written.append(out_png[:-4] + ".svg")

    labels, means, mins, maxs, meds = [], [], [], [], []
    for part in ["A", "B"]:
        for v in ["distributed", "shared"]:
            if (part, v) in rows:
                r = rows[(part, v)]
                labels.append(f"{part}-{v}")
                means.append(r.avg)
                mins.append(r.minv)
                maxs.append(r.maxv)
                meds.append(r.median)

    out_png = os.path.join(args.out, "summary_time_stats.png")
    bar_with_whiskers(
        f"{args.title_prefix} Summary runtime (bar=mean, whisker=min/max, dot=median)",
        subtitle_common, labels, means, mins, maxs, meds,
        "Time (ms)", out_png, args.svg, rotate_xticks=True
    )
    print("[OK] wrote:", out_png)
    plots_written.append(out_png)
    if args.svg:
        plots_written.append(out_png[:-4] + ".svg")

    sp_labels, sp_means, sp_meds, sp_stds = [], [], [], []
    for part in ["A", "B"]:
        if (part, "shared") in rows and (part, "distributed") in rows:
            shared = rows[(part, "shared")]
            dist = rows[(part, "distributed")]
            mean_sp, med_sp = speedup_mean_median(shared, dist)
            std_sp = speedup_std(shared, dist)

            sp_labels.append(f"Part {part}")
            sp_means.append(mean_sp)
            sp_meds.append(med_sp)
            sp_stds.append(std_sp)

    if sp_labels:
        out_png = os.path.join(args.out, "speedup_stats.png")
        speedup_plot(
            f"{args.title_prefix} Speedup (bar=mean ± std, dot=median)",
            subtitle_common, sp_labels, sp_means, sp_meds, sp_stds,
            out_png, args.svg
        )
        print("[OK] wrote:", out_png)
        plots_written.append(out_png)
        if args.svg:
            plots_written.append(out_png[:-4] + ".svg")

        for lab, m, md, sd in zip(sp_labels, sp_means, sp_meds, sp_stds):
            sd_txt = f"{sd:.3f}" if sd == sd else "nan"
            print(f"{lab}: speedup_mean={m:.3f}x  speedup_median={md:.3f}x  speedup_std~{sd_txt}x")

    rep_path = write_analysis_report(args.out, args.title_prefix, chosen, rows, plots_written)
    print("[OK] wrote:", rep_path)

    print(f"Note: parsed from: {chosen}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
