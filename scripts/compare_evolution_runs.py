"""Compare two evolution runs (e.g., h vs h_span_max).

Usage:
    python scripts/compare_evolution_runs.py \\
        --run_a rq_output/verl_ckpt_grpo_h/evolution_logs \\
        --run_b rq_output/verl_ckpt_grpo_span/evolution_logs \\
        --label_a h --label_b h_span_max \\
        --out compare_h_vs_span.md \\
        --step latest

Reads ``evo_step_<N>.json`` snapshots written by
``RQTrainer._save_evolution_snapshot`` and produces a Markdown report:

  1. Side-by-side metric summary (coverage, mean/max R_Q, hard champions, ...).
  2. Niche overlap: how many cells filled in A only / B only / both.
  3. For shared niches, the champion problem from each run plus RQ delta.
  4. Top-K champions of each run (RQ-sorted) with full problem text.

Designed to be cheap (pure JSON, no model loading).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _load_step(run_dir: Path, step: str) -> dict:
    """Load one evo_step_N.json snapshot. step='latest' picks max N."""
    files = sorted(run_dir.glob("evo_step_*.json"))
    if not files:
        raise SystemExit(f"no evo_step_*.json under {run_dir}")
    if step == "latest":
        target = files[-1]
    else:
        target = run_dir / f"evo_step_{step}.json"
        if not target.exists():
            raise SystemExit(f"{target} not found")
    return json.loads(target.read_text())


def _index_grid(snapshot: dict) -> dict[tuple[int, int], dict]:
    """{(h_bin, div_bin): cell_dict} for cells with a champion."""
    return {
        (c["h_bin"], c["div_bin"]): c
        for c in snapshot["grid"]
        if c.get("has_champion")
    }


def _short(text: str | None, n: int = 140) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def _metric_summary(snap: dict, label: str) -> list[str]:
    m = snap.get("metrics", {}) or {}
    return [
        f"### {label}",
        f"- step: {snap.get('global_step')}",
        f"- uncertainty_metric: `{snap.get('uncertainty_metric')}`",
        f"- diversity_axis: `{snap.get('diversity_axis')}`",
        f"- coverage: {m.get('grid_coverage', 0):.1%}",
        f"- champions: {m.get('grid_champions', 0)} / {m.get('total_niches', 0)}",
        f"- hard_champions (H>=H2): {m.get('hard_champions', 0)}",
        f"- mean R_Q: {m.get('grid_mean_rq', 0):.4f}",
        f"- max R_Q: {m.get('grid_max_rq', 0):.4f}",
        f"- champion p_hat: {m.get('champion_p_hat_mean', 0):.2f} ± {m.get('champion_p_hat_std', 0):.2f}",
        f"- champion H: {m.get('champion_h_mean', 0):.2f} ± {m.get('champion_h_std', 0):.2f}",
        "",
    ]


def _shared_niches_table(
    grid_a: dict, grid_b: dict, label_a: str, label_b: str,
) -> list[str]:
    shared = sorted(set(grid_a) & set(grid_b))
    if not shared:
        return ["_(no overlapping niches)_", ""]
    lines = [
        f"| niche | {label_a} R_Q | {label_b} R_Q | Δ | {label_a}: problem | {label_b}: problem |",
        "|---|---|---|---|---|---|",
    ]
    for niche in shared:
        a = grid_a[niche]
        b = grid_b[niche]
        rq_a = float(a.get("rq_score") or 0)
        rq_b = float(b.get("rq_score") or 0)
        delta = rq_b - rq_a
        lines.append(
            f"| {niche} | {rq_a:.3f} | {rq_b:.3f} | {delta:+.3f} "
            f"| {_short(a.get('problem'))} | {_short(b.get('problem'))} |"
        )
    return lines + [""]


def _exclusive_section(
    grid_only: dict, label: str,
) -> list[str]:
    if not grid_only:
        return [f"_(none — both runs covered the same niches that {label} covered)_", ""]
    lines = [f"| niche | concept | RQ | p | H | problem |", "|---|---|---|---|---|---|"]
    for niche in sorted(grid_only):
        c = grid_only[niche]
        lines.append(
            f"| {niche} | {c.get('concept_type','?')} "
            f"| {float(c.get('rq_score') or 0):.3f} "
            f"| {float(c.get('p_hat') or 0):.2f} "
            f"| {float(c.get('h_score') or 0):.2f} "
            f"| {_short(c.get('problem'))} |"
        )
    return lines + [""]


def _top_k_section(
    grid: dict, label: str, k: int,
) -> list[str]:
    cells = sorted(
        grid.values(),
        key=lambda c: -float(c.get("rq_score") or 0),
    )[:k]
    if not cells:
        return [f"_(no champions in {label})_", ""]
    lines = [f"### {label} — top {k} champions by R_Q", ""]
    for i, c in enumerate(cells, 1):
        lines += [
            f"**#{i}** niche=({c['h_bin']},{c['div_bin']}) "
            f"R_Q={float(c.get('rq_score') or 0):.4f} "
            f"p={float(c.get('p_hat') or 0):.2f} "
            f"H={float(c.get('h_score') or 0):.3f} "
            f"concept=`{c.get('concept_type','?')}`",
            "",
            "Problem:",
            "",
            "```",
            (c.get("problem") or "").strip(),
            "```",
            "",
            f"Answer: `{c.get('answer','')}`",
            "",
        ]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, type=Path)
    ap.add_argument("--run_b", required=True, type=Path)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--step", default="latest",
                    help="'latest' or an integer step number (must exist in both runs).")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("compare_evolution.md"))
    args = ap.parse_args()

    snap_a = _load_step(args.run_a, args.step)
    snap_b = _load_step(args.run_b, args.step)
    grid_a = _index_grid(snap_a)
    grid_b = _index_grid(snap_b)

    only_a = {k: v for k, v in grid_a.items() if k not in grid_b}
    only_b = {k: v for k, v in grid_b.items() if k not in grid_a}

    out = []
    out += [f"# Evolution Comparison: `{args.label_a}` vs `{args.label_b}`", ""]
    out += [f"- Run A: `{args.run_a}` step `{snap_a.get('global_step')}`",
            f"- Run B: `{args.run_b}` step `{snap_b.get('global_step')}`",
            ""]

    out += ["## 1. Metric summary", ""]
    out += _metric_summary(snap_a, args.label_a)
    out += _metric_summary(snap_b, args.label_b)

    out += ["## 2. Niche coverage overlap", ""]
    out += [f"- Filled in BOTH: **{len(set(grid_a) & set(grid_b))}**",
            f"- Filled only in `{args.label_a}`: **{len(only_a)}**",
            f"- Filled only in `{args.label_b}`: **{len(only_b)}**",
            ""]

    out += ["## 3. Shared niches — same cell, different metric", ""]
    out += _shared_niches_table(grid_a, grid_b, args.label_a, args.label_b)

    out += [f"## 4. Niches only in `{args.label_a}`", ""]
    out += _exclusive_section(only_a, args.label_a)

    out += [f"## 5. Niches only in `{args.label_b}`", ""]
    out += _exclusive_section(only_b, args.label_b)

    out += [f"## 6. Top-{args.top_k} champions per run", ""]
    out += _top_k_section(grid_a, args.label_a, args.top_k)
    out += _top_k_section(grid_b, args.label_b, args.top_k)

    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}  ({len(out)} lines)")


if __name__ == "__main__":
    main()
