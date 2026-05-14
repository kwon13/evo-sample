"""Plot per-step math-eval results from a verl checkpoint parent directory.

Reads every `<PARENT>/global_step_<N>/eval/summary.json` (pre-GPT) and, if
present, `with_gpt_judge.json` (post-GPT). Plots per-benchmark pass@1 vs step
plus the 6-benchmark macro average. Saves the figure to `<PARENT>/eval_curve.png`.

Usage:
    python scripts/plot_eval_by_step.py <PARENT_DIR> [--out OUT_PNG]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

MATH_6 = ["math500", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]
COLORS = {
    "math500":       "#1f77b4",
    "amc23":         "#ff7f0e",
    "aime24":        "#2ca02c",
    "aime25":        "#d62728",
    "minerva_math":  "#9467bd",
    "olympiadbench": "#8c564b",
}


def _step_num(d: Path) -> int:
    m = re.search(r"global_step_(\d+)$", d.name)
    return int(m.group(1)) if m else -1


def _read_summary(step_dir: Path) -> dict:
    """Return {bench: pass_at_1, 'avg6': ..., 'source': 'pre'|'gpt'} for one step.

    Prefers with_gpt_judge.json (post-GPT) if available; otherwise falls back
    to summary.json (math_verify only).
    """
    judge = step_dir / "eval" / "with_gpt_judge.json"
    summ = step_dir / "eval" / "summary.json"

    out: dict = {"bench": {}, "source": None}
    if judge.is_file():
        with judge.open() as f:
            s = json.load(f)
        for b in MATH_6:
            if b in s.get("benchmarks", {}):
                out["bench"][b] = float(s["benchmarks"][b]["pass_at_1"])
        if "math_avg_6" in s:
            out["avg6"] = float(s["math_avg_6"]) / 100.0  # stored as percent
        out["source"] = "gpt"
    elif summ.is_file():
        with summ.open() as f:
            s = json.load(f)
        for b in MATH_6:
            if b in s.get("benchmarks", {}):
                out["bench"][b] = float(s["benchmarks"][b]["pass_at_1"])
        if out["bench"]:
            out["avg6"] = sum(out["bench"].values()) / len(out["bench"])
        out["source"] = "pre"
    return out


def collect(parent: Path) -> tuple[list[int], dict[str, list[float | None]], list[float | None], list[str]]:
    step_dirs = sorted(
        [d for d in parent.glob("global_step_*") if d.is_dir() and _step_num(d) >= 0],
        key=_step_num,
    )
    steps: list[int] = []
    series: dict[str, list[float | None]] = {b: [] for b in MATH_6}
    avg6: list[float | None] = []
    sources: list[str] = []
    for d in step_dirs:
        s = _read_summary(d)
        if not s["source"]:
            continue  # no eval at all for this step
        steps.append(_step_num(d))
        for b in MATH_6:
            series[b].append(s["bench"].get(b))
        avg6.append(s.get("avg6"))
        sources.append(s["source"])
    return steps, series, avg6, sources


def _autoscale(ax, ys: list[float], pad_frac: float = 0.15, min_span: float = 4.0) -> None:
    """Set tight y-limits with padding so small movements remain visible."""
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, min_span)
    pad = span * pad_frac
    ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad))


def _draw_panel(ax, name: str, color: str, xs: list[int], ys: list[float],
                source: str | None = None) -> None:
    ax.plot(xs, ys, marker="o", lw=1.8, ms=5, color=color)
    _autoscale(ax, ys)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    if len(ys) >= 2:
        delta = ys[-1] - ys[0]
        sign = "+" if delta >= 0 else "−"
        delta_txt = f"  (Δ {sign}{abs(delta):.2f}pp)"
    else:
        delta_txt = ""
    title = f"{name}{delta_txt}"
    ax.set_title(title, fontsize=10, fontweight="bold")

    # annotate first and last
    if xs:
        ax.annotate(f"{ys[0]:.1f}", xy=(xs[0], ys[0]),
                    xytext=(-4, -10), textcoords="offset points",
                    fontsize=8, ha="right", color="gray")
        ax.annotate(f"{ys[-1]:.1f}", xy=(xs[-1], ys[-1]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color)


def plot(parent: Path, out_png: Path) -> None:
    steps, series, avg6, sources = collect(parent)
    if not steps:
        raise SystemExit(f"No evaluable step dirs under {parent}")

    pre_x = [x for x, src in zip(steps, sources) if src == "pre"]
    gpt_x = [x for x, src in zip(steps, sources) if src == "gpt"]
    if pre_x and gpt_x:
        title_src = f"mixed (pre-GPT: {len(pre_x)}, +GPT-judge: {len(gpt_x)})"
    elif gpt_x:
        title_src = "post GPT-4o judge"
    else:
        title_src = "pre GPT-4o judge (math_verify only)"

    # 2 rows × 4 cols: 6 benchmarks + avg6 + 1 unused
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    fig.suptitle(
        f"{parent.name}  —  per-step eval ({title_src})",
        fontsize=13, fontweight="bold",
    )

    panel_order = MATH_6 + ["avg6"]
    for idx, name in enumerate(panel_order):
        r, c = divmod(idx, 4)
        ax = axes[r][c]
        if name == "avg6":
            xs = [x for x, y in zip(steps, avg6) if y is not None]
            ys = [y * 100 for y in avg6 if y is not None]
            _draw_panel(ax, "MATH AVG (6)", "black", xs, ys)
            # slightly thicker line for the summary panel
            for line in ax.get_lines():
                line.set_linewidth(2.4)
                line.set_markersize(7)
        else:
            xs = [x for x, y in zip(steps, series[name]) if y is not None]
            ys = [y * 100 for y in series[name] if y is not None]
            _draw_panel(ax, name, COLORS[name], xs, ys)

        if r == 1:
            ax.set_xlabel("training step")
        if c == 0:
            ax.set_ylabel("pass@1  (%)")

    # Hide the unused cell
    axes[1][3].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"saved: {out_png}")

    # Also dump a small text table for quick reference
    print("\nstep   " + "  ".join(f"{b:>14s}" for b in MATH_6) + "  " + f"{'avg6':>6s}  src")
    for i, st in enumerate(steps):
        row = [series[b][i] for b in MATH_6]
        cells = "  ".join(f"{(v*100):>14.2f}" if v is not None else f"{'-':>14s}"
                          for v in row)
        avg_cell = f"{avg6[i]*100:>6.2f}" if avg6[i] is not None else f"{'-':>6s}"
        print(f"{st:>4d}   {cells}  {avg_cell}  {sources[i]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parent", help="verl checkpoint parent dir (contains global_step_*/)")
    p.add_argument("--out", default=None, help="output PNG path (default: <PARENT>/eval_curve.png)")
    args = p.parse_args()

    parent = Path(args.parent).resolve()
    if not parent.is_dir():
        raise SystemExit(f"not a directory: {parent}")
    out_png = Path(args.out) if args.out else parent / "eval_curve.png"
    plot(parent, out_png)


if __name__ == "__main__":
    main()
