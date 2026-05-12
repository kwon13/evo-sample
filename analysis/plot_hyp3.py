"""
Hypothesis 3 verification — does removing p_hat/h_score/rq_score from the
mutator prompt weaken evolutionary direction toward the frontier?

Inputs:
  rq_output/verl_ckpt_grpo_h_g8_/evolution_logs/  (OLD: metrics visible in prompt)
  rq_output/verl_ckpt_grpo_h_g8/evolution_logs/   (NEW: metrics stripped)

Outputs (analysis/figs/):
  hyp3_map_grid.png      — final MAP-Elites grid heatmap (OLD vs NEW), p_hat color
  hyp3_phat_hist.png     — final-champion p_hat distribution, frontier band shaded
  hyp3_timeseries.png    — over evolution_index: coverage, frontier %, p_hat mean+std
  hyp3_trajectories.png  — per-cell champion p_hat trajectory over time
"""
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/data1/yhoon113/evo-sample")
OUT  = ROOT / "analysis" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "OLD\n(metrics in prompt)": ROOT / "rq_output/verl_ckpt_grpo_h_g8_/evolution_logs",
    "NEW\n(metrics stripped)":  ROOT / "rq_output/verl_ckpt_grpo_h_g8/evolution_logs",
}

# Common reference frontier band the user is targeting.
FRONTIER_LO, FRONTIER_HI = 0.3, 0.7


def load_snapshots(d: Path):
    files = sorted(glob.glob(str(d / "evo_step_*.json")),
                   key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))
    snaps = []
    for fp in files:
        with open(fp) as f:
            snaps.append(json.load(f))
    return snaps


def grid_to_matrix(snap, n_h_bins, n_d_bins, field="p_hat"):
    """Return (n_h_bins, n_d_bins) array filled with `field` (NaN if empty)."""
    M = np.full((n_h_bins, n_d_bins), np.nan)
    for cell in snap["grid"]:
        if cell.get("has_champion"):
            v = cell.get(field, np.nan)
            M[cell["h_bin"], cell["div_bin"]] = v
    return M


def axis_dims(snap):
    g = snap["grid"]
    n_h = max(c["h_bin"] for c in g) + 1
    n_d = max(c["div_bin"] for c in g) + 1
    return n_h, n_d


def collect_timeseries(snaps):
    rows = []
    for s in snaps:
        m = s["metrics"]
        n_cham = m.get("grid_champions", 0) or 0
        front  = m.get("archive_frontier_champions", 0) or 0
        easy   = m.get("archive_too_easy_champions", 0) or 0
        hard   = m.get("archive_too_hard_champions", 0) or 0
        rows.append({
            "evo_idx":      s["evolution_index"],
            "step":         s["global_step"],
            "coverage":     m.get("grid_coverage", 0.0) or 0.0,
            "champions":    n_cham,
            "frontier":     front,
            "too_easy":     easy,
            "too_hard":     hard,
            "frontier_frac": (front / n_cham) if n_cham else 0.0,
            "p_hat_mean":   m.get("champion_p_hat_mean", np.nan),
            "p_hat_std":    m.get("champion_p_hat_std", np.nan),
            "h_mean":       m.get("champion_h_mean", np.nan),
            "h_std":        m.get("champion_h_std", np.nan),
            "accept_rate":  m.get("accept_rate", np.nan),
            "replacements": m.get("total_replacements", 0),
            "inserted":     m.get("inserted", 0),
        })
    return rows


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
runs = {label: load_snapshots(p) for label, p in RUNS.items()}
ts   = {label: collect_timeseries(s) for label, s in runs.items()}
for label, s in runs.items():
    n_h, n_d = axis_dims(s[0])
    print(f"{label!r}: n_snaps={len(s)}, grid={n_h}x{n_d}, "
          f"evo_idx {s[0]['evolution_index']}..{s[-1]['evolution_index']}")

# ---------------------------------------------------------------------------
# Figure 1: final MAP-Elites grid (p_hat heat) — OLD vs NEW
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
cmap = plt.get_cmap("RdYlGn_r")  # green near 0.5 if we center; we'll do diverging from 0.5
import matplotlib.colors as mcolors
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

for ax, (label, snaps) in zip(axes, runs.items()):
    last = snaps[-1]
    n_h, n_d = axis_dims(last)
    M = grid_to_matrix(last, n_h, n_d, "p_hat")
    seed_labels = last.get("seed_labels") or [f"d{i}" for i in range(n_d)]

    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto",
                   origin="lower", interpolation="nearest")
    # mark frontier band on the h_bin axis is irrelevant — frontier is on p_hat.
    # but we can outline cells inside the frontier band:
    for i in range(n_h):
        for j in range(n_d):
            v = M[i, j]
            if not np.isnan(v):
                in_front = FRONTIER_LO <= v <= FRONTIER_HI
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if in_front else "white",
                        weight="bold" if in_front else "normal")
                if in_front:
                    ax.add_patch(plt.Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                               fill=False, edgecolor="#1a1a1a",
                                               linewidth=1.5))
    m = last["metrics"]
    front_frac = (m.get("archive_frontier_champions", 0) or 0) / max(1, m.get("grid_champions", 1))
    ax.set_title(f"{label}\n"
                 f"evo_idx={last['evolution_index']}, "
                 f"champions={m.get('grid_champions', 0)}, "
                 f"frontier {m.get('archive_frontier_champions', 0)}/"
                 f"{m.get('grid_champions', 0)} ({front_frac:.0%})\n"
                 f"p_hat mean={m.get('champion_p_hat_mean', 0):.2f} ± "
                 f"{m.get('champion_p_hat_std', 0):.2f}",
                 fontsize=10)
    ax.set_xlabel("diversity bin (concept_group)")
    ax.set_ylabel("h_bin (uncertainty)")
    ax.set_xticks(range(n_d))
    ax.set_xticklabels(seed_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_h))

cbar = fig.colorbar(im, ax=axes, shrink=0.85,
                    label="champion p_hat (target ≈ 0.5, frontier 0.3–0.7)")
fig.suptitle("MAP-Elites final grid — cells outlined if p_hat ∈ [0.3, 0.7] (frontier)",
             fontsize=12)
fig.savefig(OUT / "hyp3_map_grid.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'hyp3_map_grid.png'}")

# ---------------------------------------------------------------------------
# Figure 2: champion p_hat histogram, frontier band shaded
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True,
                        sharey=True)
bins = np.linspace(0, 1, 21)
for ax, (label, snaps) in zip(axes, runs.items()):
    last = snaps[-1]
    p_hats = [c["p_hat"] for c in last["grid"] if c.get("has_champion")]
    ax.hist(p_hats, bins=bins, color="#3777b8", edgecolor="white")
    ax.axvspan(FRONTIER_LO, FRONTIER_HI, color="#2ca02c", alpha=0.12,
               label="frontier band")
    ax.axvline(np.mean(p_hats), color="#d62728", linestyle="--", linewidth=2,
               label=f"mean={np.mean(p_hats):.2f}")
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.5,
               label="target 0.5")
    n_front = sum(FRONTIER_LO <= p <= FRONTIER_HI for p in p_hats)
    ax.set_title(f"{label}\n"
                 f"n={len(p_hats)}, frontier {n_front}/{len(p_hats)} "
                 f"({n_front/max(1,len(p_hats)):.0%})", fontsize=10)
    ax.set_xlabel("champion p_hat")
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("count")
fig.suptitle("Final champion p_hat distribution — has the archive drifted "
             "away from the frontier?", fontsize=12)
fig.savefig(OUT / "hyp3_phat_hist.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'hyp3_phat_hist.png'}")

# ---------------------------------------------------------------------------
# Figure 3: timeseries over evolution_index
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
colors = {"OLD\n(metrics in prompt)": "#1f77b4", "NEW\n(metrics stripped)": "#d62728"}

# (a) coverage
ax = axes[0, 0]
for label, rows in ts.items():
    x = [r["evo_idx"] for r in rows]
    y = [r["coverage"] for r in rows]
    ax.plot(x, y, "o-", label=label.replace("\n"," "), color=colors[label], linewidth=2)
ax.set_xlabel("evolution_index")
ax.set_ylabel("grid coverage")
ax.set_title("Coverage (% of cells filled)")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# (b) frontier fraction
ax = axes[0, 1]
for label, rows in ts.items():
    x = [r["evo_idx"] for r in rows]
    y = [r["frontier_frac"] for r in rows]
    ax.plot(x, y, "o-", label=label.replace("\n"," "), color=colors[label], linewidth=2)
ax.axhline(1.0, color="green", linestyle=":", alpha=0.5)
ax.set_xlabel("evolution_index")
ax.set_ylabel("frontier_champions / total_champions")
ax.set_title("Fraction of champions inside frontier band [0.3, 0.7]")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# (c) p_hat mean ± std
ax = axes[1, 0]
for label, rows in ts.items():
    x = [r["evo_idx"] for r in rows]
    mean = np.array([r["p_hat_mean"] for r in rows])
    std  = np.array([r["p_hat_std"]  for r in rows])
    ax.plot(x, mean, "o-", label=label.replace("\n"," ")+" mean",
            color=colors[label], linewidth=2)
    ax.fill_between(x, mean-std, mean+std, color=colors[label], alpha=0.15)
ax.axhspan(FRONTIER_LO, FRONTIER_HI, color="#2ca02c", alpha=0.10,
           label="frontier band")
ax.axhline(0.5, color="black", linestyle=":", alpha=0.5, label="target 0.5")
ax.set_xlabel("evolution_index")
ax.set_ylabel("champion p_hat")
ax.set_title("Mean champion p_hat ± 1 std")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# (d) replacements vs inserts — wasted work indicator
ax = axes[1, 1]
for label, rows in ts.items():
    x = [r["evo_idx"] for r in rows]
    # accept_rate is per-round; cumulative ratio replacements/inserted is more telling
    repl = np.array([r["replacements"] for r in rows], dtype=float)
    ins  = np.array([r["inserted"]    for r in rows], dtype=float)
    # ratio replacements / inserted at each evo_idx
    ratio = np.divide(repl, np.maximum(ins, 1))
    ax.plot(x, ratio, "o-", label=label.replace("\n"," "),
            color=colors[label], linewidth=2)
ax.set_xlabel("evolution_index")
ax.set_ylabel("total_replacements / inserted_per_round")
ax.set_title("Replacement churn (high = mutator hits already-filled cells)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle("Hypothesis 3: weakened evolutionary direction under stripped mutator metrics",
             fontsize=13)
fig.savefig(OUT / "hyp3_timeseries.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'hyp3_timeseries.png'}")

# ---------------------------------------------------------------------------
# Figure 4: cell-level p_hat trajectory (per (h_bin, d_bin)) over time
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True,
                        sharey=True)
for ax, (label, snaps) in zip(axes, runs.items()):
    cell_traj = {}  # (h,d) -> list of (evo_idx, p_hat)
    for s in snaps:
        for c in s["grid"]:
            if c.get("has_champion"):
                key = (c["h_bin"], c["div_bin"])
                cell_traj.setdefault(key, []).append(
                    (s["evolution_index"], c["p_hat"])
                )
    for key, pts in cell_traj.items():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", alpha=0.35, linewidth=1)
    ax.axhspan(FRONTIER_LO, FRONTIER_HI, color="#2ca02c", alpha=0.10)
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("evolution_index")
    ax.set_title(label.replace("\n", " "))
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("champion p_hat per (h_bin, d_bin)")
fig.suptitle("Per-cell champion p_hat trajectories (each line = one MAP-Elites cell)",
             fontsize=12)
fig.savefig(OUT / "hyp3_trajectories.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT/'hyp3_trajectories.png'}")

# ---------------------------------------------------------------------------
# Side print: per-label summary numbers
# ---------------------------------------------------------------------------
print("\n=== Per-iteration p_hat summary ===")
print(f"{'evo_idx':>8s}  {'cov':>5s}  {'cham':>5s}  {'front':>5s}  "
      f"{'easy':>5s}  {'hard':>5s}  {'mean':>5s}  {'std':>5s}  "
      f"{'repl':>5s}")
for label, rows in ts.items():
    print(f"\n-- {label.replace(chr(10),' ')} --")
    for r in rows[::max(1, len(rows)//8)]:
        print(f"{r['evo_idx']:>8d}  {r['coverage']:>5.2f}  "
              f"{r['champions']:>5d}  {r['frontier']:>5d}  "
              f"{r['too_easy']:>5d}  {r['too_hard']:>5d}  "
              f"{r['p_hat_mean']:>5.2f}  {r['p_hat_std']:>5.2f}  "
              f"{r['replacements']:>5d}")
