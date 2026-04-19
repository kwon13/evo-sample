"""
Visualize RQ-Evolve evolution logs as MAP-Elites maps.

This script reads:
  - evolution_logs/evo_step_*.json or evolution_logs/latest.json
  - evolution_logs/datasets/dataset_step_*.json or latest_dataset.json

and writes a self-contained HTML dashboard with two maps per step:
  - Archive map: champion R_Q / p_hat / H over H x D niches
  - Dataset map: training examples emitted from each H x D niche

Usage:
  python scripts/visualize_evolution_map.py \
      --log_dir ./rq_output/verl_ckpt/evolution_logs \
      --output ./rq_output/evolution_map.html

  # With downloaded files where evo_step_*.json and datasets/ live separately:
  python scripts/visualize_evolution_map.py \
      --log_dir /Users/kyhoon13/Downloads \
      --dataset_dir /Users/kyhoon13/Downloads/datasets \
      --output /tmp/evolution_map.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _step_from_path(path: Path) -> int:
    match = re.search(r"(?:evo|dataset)_step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return 10**12


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_evolution_snapshots(log_dir: Path) -> list[dict[str, Any]]:
    files = sorted(log_dir.glob("evo_step_*.json"), key=_step_from_path)
    if files:
        return [_load_json(path) | {"_source_file": str(path)} for path in files]

    latest = log_dir / "latest.json"
    if latest.exists():
        return [_load_json(latest) | {"_source_file": str(latest)}]

    raise FileNotFoundError(
        f"No evo_step_*.json or latest.json found under {log_dir}"
    )


def load_dataset_snapshots(dataset_dir: Path | None) -> dict[int, dict[str, Any]]:
    if dataset_dir is None or not dataset_dir.exists():
        return {}

    files = sorted(dataset_dir.glob("dataset_step_*.json"), key=_step_from_path)
    if not files:
        latest = dataset_dir / "latest_dataset.json"
        files = [latest] if latest.exists() else []

    datasets: dict[int, dict[str, Any]] = {}
    for path in files:
        snap = _load_json(path)
        step = int(snap.get("global_step", _step_from_path(path)))
        snap["_source_file"] = str(path)
        datasets[step] = snap
    return datasets


def _grid_shape(snapshots: list[dict[str, Any]]) -> tuple[int, int]:
    max_h = max(cell["h_bin"] for snap in snapshots for cell in snap["grid"])
    max_d = max(cell["div_bin"] for snap in snapshots for cell in snap["grid"])
    return max_h + 1, max_d + 1


def _summarize_dataset(dataset: dict[str, Any] | None) -> dict[str, Any]:
    if not dataset:
        return {"cells": [], "programs": [], "problem_count": 0}

    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    by_program: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for problem in dataset.get("problems", []):
        h = problem.get("h_bin")
        d = problem.get("d_bin")
        if h is not None and d is not None:
            by_cell[(int(h), int(d))].append(problem)
        by_program[str(problem.get("program_id", "<unknown>"))].append(problem)

    cells = []
    for (h, d), problems in sorted(by_cell.items()):
        programs = sorted({str(p.get("program_id", "<unknown>")) for p in problems})
        seeds = sorted(p.get("seed") for p in problems if p.get("seed") is not None)
        cells.append({
            "h": h,
            "d": d,
            "count": len(problems),
            "programs": programs,
            "seed_min": min(seeds) if seeds else None,
            "seed_max": max(seeds) if seeds else None,
            "examples": problems[:3],
        })

    programs = []
    for program_id, problems in sorted(
        by_program.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        seeds = sorted(p.get("seed") for p in problems if p.get("seed") is not None)
        cells_for_program = sorted({
            f"H{p.get('h_bin')} D{p.get('d_bin')}"
            for p in problems
            if p.get("h_bin") is not None and p.get("d_bin") is not None
        })
        programs.append({
            "program_id": program_id,
            "count": len(problems),
            "seed_min": min(seeds) if seeds else None,
            "seed_max": max(seeds) if seeds else None,
            "cells": cells_for_program,
        })

    return {
        "cells": cells,
        "programs": programs,
        "problem_count": len(dataset.get("problems", [])),
    }


def build_payload(
    snapshots: list[dict[str, Any]],
    datasets_by_step: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    n_h, n_d = _grid_shape(snapshots)
    payload_steps = []

    for index, snap in enumerate(snapshots):
        step = int(snap.get("global_step", index))
        dataset = datasets_by_step.get(step)
        grid_cells = []
        for cell in snap["grid"]:
            has = bool(cell.get("has_champion"))
            grid_cells.append({
                "h": int(cell["h_bin"]),
                "d": int(cell["div_bin"]),
                "has": has,
                "rq": float(cell.get("rq_score") or 0.0) if has else 0.0,
                "p": cell.get("p_hat") if has else None,
                "H": cell.get("h_score") if has else None,
                "generation": cell.get("generation") if has else None,
                "program_id": cell.get("program_id") if has else None,
                "problem": cell.get("problem", "") if has else "",
                "answer": cell.get("answer", "") if has else "",
            })

        metrics = snap.get("metrics", {})
        ds_summary = _summarize_dataset(dataset)
        payload_steps.append({
            "index": index,
            "global_step": step,
            "source_file": snap.get("_source_file", ""),
            "dataset_file": dataset.get("_source_file", "") if dataset else "",
            "metrics": metrics,
            "grid": grid_cells,
            "dataset": ds_summary,
            "dataset_meta": {
                "dataset_size": dataset.get("dataset_size", 0) if dataset else 0,
                "archive_champions": dataset.get("archive_champions", None)
                if dataset else None,
                "frontier_champions": dataset.get("frontier_champions", None)
                if dataset else None,
                "instances_per_program": dataset.get("instances_per_program", None)
                if dataset else None,
                "strict_anti_reuse": dataset.get("strict_anti_reuse", None)
                if dataset else None,
            },
        })

    return {
        "n_h": n_h,
        "n_d": n_d,
        "steps": payload_steps,
    }


def write_html(payload: dict[str, Any], output_path: Path) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>RQ-Evolve MAP-Elites Map</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #101418; color: #eef2f4; }}
main {{ padding: 24px; max-width: 1480px; margin: 0 auto; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 28px 0 12px; font-size: 18px; color: #9dd7c8; }}
.muted {{ color: #93a1a1; }}
.toolbar {{ display: flex; gap: 12px; align-items: center; margin: 18px 0; flex-wrap: wrap; }}
input[type=range] {{ width: min(620px, 100%); accent-color: #42c6a5; }}
button {{ border: 1px solid #42c6a5; background: #182028; color: #eef2f4; border-radius: 6px; padding: 7px 12px; cursor: pointer; }}
button:hover {{ background: #24313a; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0 22px; }}
.card {{ border: 1px solid #26333d; background: #151c22; border-radius: 8px; padding: 12px; }}
.card .label {{ color: #93a1a1; font-size: 12px; }}
.card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
.maps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; align-items: start; }}
.panel {{ border: 1px solid #26333d; background: #151c22; border-radius: 8px; padding: 14px; overflow-x: auto; }}
table.map {{ border-collapse: collapse; font-size: 12px; }}
table.map th {{ background: #202a32; color: #b9c7c7; padding: 7px; border: 1px solid #11171c; white-space: nowrap; }}
table.map td {{ width: 74px; height: 46px; text-align: center; border: 1px solid #11171c; cursor: default; position: relative; }}
td.empty {{ background: #0d1115; color: #303b42; }}
td.cell:hover {{ outline: 2px solid #e7d37f; z-index: 2; }}
.row-label {{ background: #202a32; color: #b9c7c7; width: 52px; position: sticky; left: 0; z-index: 1; }}
.legend {{ display: flex; gap: 16px; flex-wrap: wrap; color: #93a1a1; font-size: 12px; margin: 8px 0 12px; }}
.tip {{ position: fixed; display: none; max-width: 520px; background: #101418; color: #eef2f4; border: 1px solid #42c6a5; border-radius: 8px; padding: 12px; font-size: 12px; line-height: 1.5; white-space: pre-wrap; z-index: 1000; box-shadow: 0 10px 30px rgba(0,0,0,.35); }}
.tables {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
table.list {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
table.list th, table.list td {{ border-bottom: 1px solid #26333d; padding: 8px; text-align: left; vertical-align: top; }}
code {{ color: #9dd7c8; }}
</style>
</head>
<body>
<main>
  <h1>RQ-Evolve MAP-Elites Map</h1>
  <div class="muted">Archive grid와 실제 training dataset 분포를 같은 step에서 비교합니다.</div>
  <div class="toolbar">
    <span>Step</span>
    <input id="slider" type="range" min="0" max="0" value="0">
    <strong id="stepLabel"></strong>
    <button id="play">Play</button>
  </div>
  <div class="cards" id="cards"></div>
  <div class="maps">
    <section class="panel">
      <h2>Archive Map: Champion R_Q</h2>
      <div class="legend">색 진할수록 R_Q 높음 · 빈 칸은 champion 없음 · hover로 문제/답 확인</div>
      <div id="archiveMap"></div>
    </section>
    <section class="panel">
      <h2>Dataset Map: Training Examples</h2>
      <div class="legend">색 진할수록 해당 niche에서 학습 문제가 많이 생성됨 · hover로 program/seed 확인</div>
      <div id="datasetMap"></div>
    </section>
  </div>
  <h2>Dataset Programs</h2>
  <div class="panel" id="programTable"></div>
  <h2>Files</h2>
  <div class="panel muted" id="files"></div>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = {payload_json};
const slider = document.getElementById('slider');
const label = document.getElementById('stepLabel');
const tip = document.getElementById('tip');
slider.max = Math.max(DATA.steps.length - 1, 0);
slider.value = DATA.steps.length - 1;

function fmt(x, digits=3) {{
  if (x === null || x === undefined || Number.isNaN(Number(x))) return '·';
  return Number(x).toFixed(digits);
}}
function esc(s) {{
  return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function rqColor(rq, maxRq) {{
  if (!rq || maxRq <= 0) return '#0d1115';
  const t = Math.max(0, Math.min(1, rq / maxRq));
  const r = Math.round(20 + 58 * t);
  const g = Math.round(28 + 170 * t);
  const b = Math.round(34 + 135 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function countColor(count, maxCount) {{
  if (!count || maxCount <= 0) return '#0d1115';
  const t = Math.max(0, Math.min(1, count / maxCount));
  const r = Math.round(30 + 215 * t);
  const g = Math.round(36 + 164 * t);
  const b = Math.round(42 + 50 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function makeTable(cells, valueFn, colorFn, tipFn) {{
  let html = '<table class="map"><tr><th>D \\\\ H</th>';
  for (let h = 0; h < DATA.n_h; h++) html += `<th>H${{h}}</th>`;
  html += '</tr>';
  for (let d = 0; d < DATA.n_d; d++) {{
    html += `<tr><th class="row-label">D${{d}}</th>`;
    for (let h = 0; h < DATA.n_h; h++) {{
      const cell = cells.find(c => c.h === h && c.d === d);
      if (!cell) {{
        html += '<td class="empty">·</td>';
      }} else {{
        html += `<td class="cell" style="background:${{colorFn(cell)}}" data-tip="${{esc(tipFn(cell))}}">${{valueFn(cell)}}</td>`;
      }}
    }}
    html += '</tr>';
  }}
  html += '</table>';
  return html;
}}
function render(index) {{
  const step = DATA.steps[index];
  const metrics = step.metrics || {{}};
  const meta = step.dataset_meta || {{}};
  label.textContent = `index ${{index}} · global_step ${{step.global_step}}`;
  const maxRq = Math.max(...step.grid.map(c => c.rq || 0), 0.000001);
  const dsCells = step.dataset.cells || [];
  const maxCount = Math.max(...dsCells.map(c => c.count || 0), 1);
  document.getElementById('cards').innerHTML = [
    ['grid champions', metrics.grid_champions],
    ['frontier champions', metrics.archive_frontier_champions ?? meta.frontier_champions],
    ['dataset size', meta.dataset_size ?? step.dataset.problem_count],
    ['coverage', fmt(metrics.grid_coverage, 3)],
    ['accept rate', fmt(metrics.accept_rate, 3)],
    ['max R_Q', fmt(metrics.grid_max_rq, 4)],
    ['low-H evicted', metrics.reeval_low_h_evicted],
    ['inserted', metrics.inserted],
  ].map(([k,v]) => `<div class="card"><div class="label">${{k}}</div><div class="value">${{v ?? '·'}}</div></div>`).join('');
  document.getElementById('archiveMap').innerHTML = makeTable(
    step.grid.filter(c => c.has),
    c => fmt(c.rq, 3),
    c => rqColor(c.rq, maxRq),
    c => `H${{c.h}} D${{c.d}}\\nprogram=${{c.program_id}}\\nRQ=${{fmt(c.rq, 5)}} p=${{fmt(c.p, 3)}} H=${{fmt(c.H, 3)}} gen=${{c.generation}}\\n\\nQ: ${{c.problem}}\\nA: ${{c.answer}}`
  );
  document.getElementById('datasetMap').innerHTML = makeTable(
    dsCells,
    c => c.count,
    c => countColor(c.count, maxCount),
    c => `H${{c.h}} D${{c.d}}\\ncount=${{c.count}}\\nprograms=${{c.programs.join(', ')}}\\nseeds=${{c.seed_min}}..${{c.seed_max}}\\n\\n` + (c.examples || []).map(p => `seed ${{p.seed}} · ${{p.program_id}}\\nQ: ${{p.problem}}\\nA: ${{p.answer}}`).join('\\n\\n')
  );
  const programs = step.dataset.programs || [];
  document.getElementById('programTable').innerHTML = programs.length ? (
    '<table class="list"><tr><th>program_id</th><th>count</th><th>seed range</th><th>cells</th></tr>' +
    programs.map(p => `<tr><td><code>${{esc(p.program_id)}}</code></td><td>${{p.count}}</td><td>${{p.seed_min}}..${{p.seed_max}}</td><td>${{esc((p.cells || []).join(', '))}}</td></tr>`).join('') +
    '</table>'
  ) : '<span class="muted">No dataset examples for this step.</span>';
  document.getElementById('files').innerHTML =
    `evolution: <code>${{esc(step.source_file)}}</code><br>` +
    `dataset: <code>${{esc(step.dataset_file || 'not found')}}</code>`;
}}
slider.addEventListener('input', () => render(Number(slider.value)));
document.getElementById('play').addEventListener('click', () => {{
  let i = 0;
  const timer = setInterval(() => {{
    slider.value = i;
    render(i);
    i += 1;
    if (i >= DATA.steps.length) clearInterval(timer);
  }}, 900);
}});
document.body.addEventListener('mouseover', e => {{
  const text = e.target?.getAttribute?.('data-tip');
  if (!text) return;
  tip.textContent = text;
  tip.style.display = 'block';
}});
document.body.addEventListener('mousemove', e => {{
  tip.style.left = `${{e.clientX + 14}}px`;
  tip.style.top = `${{e.clientY + 14}}px`;
}});
document.body.addEventListener('mouseout', e => {{
  if (e.target?.getAttribute?.('data-tip')) tip.style.display = 'none';
}});
render(Number(slider.value));
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def print_summary(payload: dict[str, Any], output_path: Path) -> None:
    latest = payload["steps"][-1]
    programs = latest["dataset"].get("programs", [])
    print(f"Wrote {output_path}")
    print(
        f"Loaded {len(payload['steps'])} evolution step(s); "
        f"latest global_step={latest['global_step']}"
    )
    print(
        "Latest: "
        f"grid_champions={latest['metrics'].get('grid_champions')}, "
        f"frontier={latest['metrics'].get('archive_frontier_champions')}, "
        f"dataset_size={latest['dataset_meta'].get('dataset_size')}"
    )
    if programs:
        counts = Counter({p["program_id"]: p["count"] for p in programs})
        top = ", ".join(f"{pid}:{count}" for pid, count in counts.most_common(5))
        print(f"Top dataset programs: {top}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained MAP-Elites map from evolution logs."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("./rq_output/verl_ckpt/evolution_logs"),
        help="Directory containing evo_step_*.json or latest.json",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Directory containing dataset_step_*.json. Defaults to log_dir/datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./rq_output/evolution_map.html"),
        help="Output HTML path.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or (args.log_dir / "datasets")
    snapshots = load_evolution_snapshots(args.log_dir)
    datasets_by_step = load_dataset_snapshots(dataset_dir)
    payload = build_payload(snapshots, datasets_by_step)
    write_html(payload, args.output)
    print_summary(payload, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"visualize_evolution_map.py: {exc}", file=sys.stderr)
        sys.exit(1)
