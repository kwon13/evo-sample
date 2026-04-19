"""
Visualize RQ-Evolve evolution logs as MAP-Elites maps.

This script reads:
  - evolution_logs/evo_step_*.json or evolution_logs/latest.json
  - evolution_logs/datasets/dataset_step_*.json or latest_dataset.json
  - evolution_logs/events/events_evo_*_step_*.jsonl

and writes a self-contained HTML dashboard with three maps per step:
  - Archive map: champion R_Q / p_hat / H over H x D niches
  - Dataset map: training examples emitted from each H x D niche
  - Candidate event map: mutation/crossover candidates as they are scored

Usage:
  python scripts/visualize_evolution_map.py \
      --log_dir ./rq_output/verl_ckpt/evolution_logs \
      --output ./rq_output/evolution_map.html

  # Live-ish monitoring while training is running. Keep this process open,
  # then open/refresh the generated HTML in a browser.
  python scripts/visualize_evolution_map.py \
      --log_dir ./rq_output/verl_ckpt/evolution_logs \
      --output ./rq_output/evolution_map.html \
      --watch_interval 2

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


def _event_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r"events_evo_(\d+)_step_(\d+)", path.stem)
    if match:
        evo_idx = int(match.group(1))
        step = int(match.group(2))
        return (step, evo_idx, path.name)
    return (10**12, 10**12, path.name)


def load_event_streams(events_dir: Path | None) -> list[dict[str, Any]]:
    """Load append-only candidate event logs produced during evolution."""
    if events_dir is None or not events_dir.exists():
        return []

    files = sorted(events_dir.glob("events_evo_*_step_*.jsonl"), key=_event_sort_key)
    if not files:
        latest = events_dir / "latest_events.jsonl"
        files = [latest] if latest.exists() else []

    streams = []
    for path in files:
        events = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"Skipping malformed event line {path}:{line_no}: {exc}",
                        file=sys.stderr,
                    )
        if not events:
            continue

        header = events[0] if events[0].get("event") == "evolution_start" else {}
        match = re.search(r"events_evo_(\d+)_step_(\d+)", path.stem)
        evo_idx = header.get("evolution_index")
        step = header.get("global_step")
        if match:
            evo_idx = evo_idx if evo_idx is not None else int(match.group(1))
            step = step if step is not None else int(match.group(2))
        streams.append({
            "source_file": str(path),
            "global_step": int(step) if step is not None else _event_sort_key(path)[0],
            "evolution_index": int(evo_idx) if evo_idx is not None else None,
            "events": events,
        })
    return streams


def _grid_shape(
    snapshots: list[dict[str, Any]],
    event_streams: list[dict[str, Any]] | None = None,
    default_shape: tuple[int, int] = (6, 10),
) -> tuple[int, int]:
    if snapshots:
        max_h = max(cell["h_bin"] for snap in snapshots for cell in snap["grid"])
        max_d = max(cell["div_bin"] for snap in snapshots for cell in snap["grid"])
        return max_h + 1, max_d + 1

    h_vals = []
    d_vals = []
    for stream in event_streams or []:
        for event in stream.get("events", []):
            if event.get("h_bin") is not None:
                h_vals.append(int(event["h_bin"]))
            if event.get("d_bin") is not None:
                d_vals.append(int(event["d_bin"]))
    if h_vals and d_vals:
        return max(max(h_vals) + 1, default_shape[0]), max(
            max(d_vals) + 1, default_shape[1]
        )
    return default_shape


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


def _short_id(value: Any) -> str:
    text = str(value or "")
    return text[:10] if len(text) > 10 else text


def _summarize_events(streams: list[dict[str, Any]]) -> dict[str, Any]:
    """Compress event JSONL streams into a browser-friendly timeline."""
    raw_events: list[dict[str, Any]] = []
    source_files = []
    for stream in streams:
        source_files.append(stream.get("source_file", ""))
        raw_events.extend(stream.get("events", []))

    raw_events.sort(key=lambda e: (
        int(e.get("evolution_index") or 0),
        int(e.get("event_seq") or 0),
    ))

    timeline = []
    for event in raw_events:
        event_type = event.get("event", "")
        if event_type == "evolution_start":
            continue
        timeline.append({
            "seq": int(event.get("event_seq") or len(timeline) + 1),
            "round": event.get("round"),
            "event": event_type,
            "status": event.get("status"),
            "op": event.get("op"),
            "child_id": event.get("child_id"),
            "child_short": _short_id(event.get("child_id")),
            "parent_id": event.get("parent_id"),
            "parent_short": _short_id(event.get("parent_id")),
            "parent_b_id": event.get("parent_b_id"),
            "h": event.get("h_bin"),
            "d": event.get("d_bin"),
            "rq": event.get("rq_score"),
            "p": event.get("p_hat"),
            "H": event.get("h_bar"),
            "frontier_status": event.get("frontier_status"),
            "reservoir_saved": event.get("reservoir_saved"),
            "reservoir_reason": event.get("reservoir_reason"),
            "previous_program_id": event.get("previous_program_id"),
            "previous_short": _short_id(event.get("previous_program_id")),
            "previous_rq": event.get("previous_rq"),
            "num_rollouts": event.get("num_rollouts"),
            "num_correct": event.get("num_correct"),
            "error": event.get("error"),
            "reason": event.get("reason"),
            "problem": event.get("problem", ""),
            "answer": event.get("answer", ""),
        })

    status_counts = Counter(
        e["status"] for e in timeline if e.get("status") is not None
    )
    event_counts = Counter(e["event"] for e in timeline)
    op_counts = Counter(e["op"] for e in timeline if e.get("op") is not None)

    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in timeline:
        h, d = event.get("h"), event.get("d")
        if h is None or d is None:
            continue
        by_cell[(int(h), int(d))].append(event)

    cells = []
    for (h, d), events in sorted(by_cell.items()):
        cell_status = Counter(
            e["status"] for e in events if e.get("status") is not None
        )
        cell_ops = Counter(e["op"] for e in events if e.get("op") is not None)
        cells.append({
            "h": h,
            "d": d,
            "count": len(events),
            "inserted": int(cell_status.get("inserted", 0)),
            "rejected": int(cell_status.get("rejected", 0)),
            "h_skipped": int(cell_status.get("h_prefilter_skip", 0)),
            "ops": dict(cell_ops),
            "last_events": events[-6:],
        })

    return {
        "event_count": len(timeline),
        "status_counts": dict(status_counts),
        "event_counts": dict(event_counts),
        "op_counts": dict(op_counts),
        "cells": cells,
        "timeline": timeline,
        "source_files": source_files,
    }


def build_payload(
    snapshots: list[dict[str, Any]],
    datasets_by_step: dict[int, dict[str, Any]],
    event_streams: list[dict[str, Any]] | None = None,
    default_shape: tuple[int, int] = (6, 10),
) -> dict[str, Any]:
    n_h, n_d = _grid_shape(snapshots, event_streams, default_shape)
    payload_steps = []
    snapshot_steps = set()
    event_streams_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for stream in event_streams or []:
        event_streams_by_step[int(stream.get("global_step", 10**12))].append(stream)

    for index, snap in enumerate(snapshots):
        step = int(snap.get("global_step", index))
        snapshot_steps.add(step)
        dataset = datasets_by_step.get(step)
        event_summary = _summarize_events(event_streams_by_step.get(step, []))
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
                "reservoir_count": int(cell.get("reservoir_count", 0) or 0),
                "reservoir_best_rq": float(cell.get("reservoir_best_rq", 0.0) or 0.0),
                "reservoir_program_ids": cell.get("reservoir_program_ids", []),
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
            "events": event_summary,
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

    # When an evolution is currently running, candidate JSONL events are written
    # before the end-of-step snapshot exists. Keep those event-only steps visible
    # so the dashboard can be used while evolution is in flight.
    for step in sorted(set(event_streams_by_step) - snapshot_steps):
        event_summary = _summarize_events(event_streams_by_step[step])
        payload_steps.append({
            "index": len(payload_steps),
            "global_step": step,
            "source_file": "",
            "dataset_file": "",
            "metrics": {},
            "grid": [
                {
                    "h": h,
                    "d": d,
                    "has": False,
                    "rq": 0.0,
                    "p": None,
                    "H": None,
                    "generation": None,
                    "program_id": None,
                    "reservoir_count": 0,
                    "reservoir_best_rq": 0.0,
                    "reservoir_program_ids": [],
                    "problem": "",
                    "answer": "",
                }
                for d in range(n_d)
                for h in range(n_h)
            ],
            "dataset": {"cells": [], "programs": [], "problem_count": 0},
            "events": event_summary,
            "dataset_meta": {
                "dataset_size": 0,
                "archive_champions": None,
                "frontier_champions": None,
                "instances_per_program": None,
                "strict_anti_reuse": None,
            },
        })

    payload_steps.sort(key=lambda s: (int(s.get("global_step", 10**12)), s["index"]))
    for index, step_payload in enumerate(payload_steps):
        step_payload["index"] = index

    return {
        "n_h": n_h,
        "n_d": n_d,
        "steps": payload_steps,
    }


def write_html(
    payload: dict[str, Any], output_path: Path, auto_refresh_seconds: float = 0.0,
) -> None:
    payload_json = json.dumps(payload, ensure_ascii=False)
    refresh_js = (
        f"setTimeout(() => location.reload(), {int(auto_refresh_seconds * 1000)});"
        if auto_refresh_seconds and auto_refresh_seconds > 0
        else ""
    )
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
.status {{ font-weight: 700; }}
.status-inserted {{ color: #9dd7c8; }}
.status-rejected {{ color: #f0a08b; }}
.status-h {{ color: #e7d37f; }}
.status-failed {{ color: #c9a7ff; }}
code {{ color: #9dd7c8; }}
</style>
</head>
<body>
<main>
  <h1>RQ-Evolve MAP-Elites Map</h1>
  <div class="muted">Archive snapshot, training dataset, candidate mutation/crossover events를 같은 지도 위에서 비교합니다.</div>
  <div class="toolbar">
    <span>Step</span>
    <input id="slider" type="range" min="0" max="0" value="0">
    <strong id="stepLabel"></strong>
    <button id="play">Play</button>
  </div>
  <div class="toolbar">
    <span>Event playback</span>
    <input id="eventSlider" type="range" min="0" max="0" value="0">
    <strong id="eventLabel"></strong>
    <button id="playEvents">Play Events</button>
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
    <section class="panel">
      <h2>Candidate Event Map</h2>
      <div class="legend">event slider를 움직이면 mutation/crossover 후보가 어느 셀에 scored, inserted, rejected 되었는지 순서대로 누적됩니다.</div>
      <div id="eventMap"></div>
    </section>
  </div>
  <h2>Candidate Timeline</h2>
  <div class="panel" id="eventTable"></div>
  <h2>Dataset Programs</h2>
  <div class="panel" id="programTable"></div>
  <h2>Files</h2>
  <div class="panel muted" id="files"></div>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = {payload_json};
const slider = document.getElementById('slider');
const eventSlider = document.getElementById('eventSlider');
const label = document.getElementById('stepLabel');
const eventLabel = document.getElementById('eventLabel');
const tip = document.getElementById('tip');
slider.max = Math.max(DATA.steps.length - 1, 0);
slider.value = DATA.steps.length - 1;
let currentStepIndex = Number(slider.value);

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
function eventColor(cell, maxCount) {{
  const t = Math.max(0.25, Math.min(1, (cell.count || 0) / Math.max(maxCount, 1)));
  if ((cell.inserted || 0) > 0) {{
    return `rgb(${{Math.round(22 + 50 * t)}},${{Math.round(92 + 150 * t)}},${{Math.round(76 + 95 * t)}})`;
  }}
  if ((cell.h_skipped || 0) > 0 && (cell.rejected || 0) === 0) {{
    return `rgb(${{Math.round(85 + 135 * t)}},${{Math.round(74 + 118 * t)}},${{Math.round(32 + 40 * t)}})`;
  }}
  if ((cell.rejected || 0) > 0) {{
    return `rgb(${{Math.round(92 + 150 * t)}},${{Math.round(58 + 62 * t)}},${{Math.round(50 + 40 * t)}})`;
  }}
  return countColor(cell.count, maxCount);
}}
function statusClass(status, eventName='') {{
  if (status === 'inserted') return 'status status-inserted';
  if (status === 'rejected') return 'status status-rejected';
  if (status === 'h_prefilter_skip') return 'status status-h';
  if (String(eventName).includes('failed')) return 'status status-failed';
  return 'status';
}}
function aggregateEvents(step, limit) {{
  const timeline = (step.events?.timeline || []).slice(0, limit);
  const cellsByKey = new Map();
  const statusCounts = {{}};
  const eventCounts = {{}};
  for (const ev of timeline) {{
    if (ev.status) statusCounts[ev.status] = (statusCounts[ev.status] || 0) + 1;
    eventCounts[ev.event] = (eventCounts[ev.event] || 0) + 1;
    if (ev.h === null || ev.h === undefined || ev.d === null || ev.d === undefined) continue;
    const key = `${{ev.h}}:${{ev.d}}`;
    if (!cellsByKey.has(key)) {{
      cellsByKey.set(key, {{
        h: Number(ev.h), d: Number(ev.d), count: 0,
        inserted: 0, rejected: 0, h_skipped: 0, ops: {{}}, last_events: []
      }});
    }}
    const cell = cellsByKey.get(key);
    cell.count += 1;
    if (ev.status === 'inserted') cell.inserted += 1;
    if (ev.status === 'rejected') cell.rejected += 1;
    if (ev.status === 'h_prefilter_skip') cell.h_skipped += 1;
    if (ev.op) cell.ops[ev.op] = (cell.ops[ev.op] || 0) + 1;
    cell.last_events.push(ev);
    if (cell.last_events.length > 6) cell.last_events.shift();
  }}
  return {{
    timeline,
    cells: Array.from(cellsByKey.values()),
    statusCounts,
    eventCounts,
  }};
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
function render(index, resetEvents=true) {{
  currentStepIndex = index;
  const step = DATA.steps[index];
  const totalEvents = step.events?.timeline?.length || 0;
  eventSlider.max = totalEvents;
  if (resetEvents || Number(eventSlider.value) > totalEvents) eventSlider.value = totalEvents;
  renderStep(Number(eventSlider.value));
}}
function renderStep(eventLimit) {{
  const step = DATA.steps[currentStepIndex];
  const metrics = step.metrics || {{}};
  const meta = step.dataset_meta || {{}};
  const totalEvents = step.events?.timeline?.length || 0;
  const ev = aggregateEvents(step, eventLimit);
  label.textContent = `index ${{currentStepIndex}} · global_step ${{step.global_step}}`;
  eventLabel.textContent = `${{eventLimit}}/${{totalEvents}}`;
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
    ['reservoir', metrics.reservoir_candidates],
    ['reservoir selections', metrics.reservoir_selections],
    ['events shown', `${{ev.timeline.length}}/${{totalEvents}}`],
    ['verified', ev.eventCounts.candidate_verified],
    ['event inserted', ev.statusCounts.inserted],
    ['event rejected', ev.statusCounts.rejected],
    ['H skipped', ev.statusCounts.h_prefilter_skip],
  ].map(([k,v]) => `<div class="card"><div class="label">${{k}}</div><div class="value">${{v ?? '·'}}</div></div>`).join('');
  document.getElementById('archiveMap').innerHTML = makeTable(
    step.grid.filter(c => c.has),
    c => fmt(c.rq, 3),
    c => rqColor(c.rq, maxRq),
    c => `H${{c.h}} D${{c.d}}\\nprogram=${{c.program_id}}\\nRQ=${{fmt(c.rq, 5)}} p=${{fmt(c.p, 3)}} H=${{fmt(c.H, 3)}} gen=${{c.generation}}\\nreservoir=${{c.reservoir_count || 0}} best=${{fmt(c.reservoir_best_rq, 4)}} ids=${{(c.reservoir_program_ids || []).join(', ') || '·'}}\\n\\nQ: ${{c.problem}}\\nA: ${{c.answer}}`
  );
  document.getElementById('datasetMap').innerHTML = makeTable(
    dsCells,
    c => c.count,
    c => countColor(c.count, maxCount),
    c => `H${{c.h}} D${{c.d}}\\ncount=${{c.count}}\\nprograms=${{c.programs.join(', ')}}\\nseeds=${{c.seed_min}}..${{c.seed_max}}\\n\\n` + (c.examples || []).map(p => `seed ${{p.seed}} · ${{p.program_id}}\\nQ: ${{p.problem}}\\nA: ${{p.answer}}`).join('\\n\\n')
  );
  const evCells = ev.cells || [];
  const maxEventCount = Math.max(...evCells.map(c => c.count || 0), 1);
  document.getElementById('eventMap').innerHTML = totalEvents ? makeTable(
    evCells,
    c => `${{c.inserted}}/${{c.count}}`,
    c => eventColor(c, maxEventCount),
    c => `H${{c.h}} D${{c.d}}\\nevents=${{c.count}} inserted=${{c.inserted}} rejected=${{c.rejected}} H-skip=${{c.h_skipped}}\\nops=${{Object.entries(c.ops || {{}}).map(([k,v]) => `${{k}}:${{v}}`).join(', ') || '·'}}\\n\\n` +
      (c.last_events || []).map(e => `#${{e.seq}} r${{e.round ?? '·'}} ${{e.op || e.event}} ${{e.status || ''}}${{e.reservoir_saved ? ' reservoir' : ''}}\\nchild=${{e.child_short || '·'}} parent=${{e.parent_short || '·'}}\\nRQ=${{fmt(e.rq, 4)}} p=${{fmt(e.p, 3)}} H=${{fmt(e.H, 3)}}\\nQ: ${{e.problem || ''}}`).join('\\n\\n')
  ) : '<span class="muted">No candidate event log for this step.</span>';
  const rows = ev.timeline.slice(-140);
  document.getElementById('eventTable').innerHTML = rows.length ? (
    `<div class="muted">Showing last ${{rows.length}} of ${{ev.timeline.length}} visible events. Move the event slider to replay scoring order.</div>` +
    '<table class="list"><tr><th>#</th><th>round</th><th>op</th><th>status</th><th>cell</th><th>RQ / p / H</th><th>parent → child</th><th>reservoir</th><th>previous</th><th>problem</th></tr>' +
    rows.map(e => `<tr>` +
      `<td>${{e.seq}}</td>` +
      `<td>${{e.round ?? '·'}}</td>` +
      `<td>${{esc(e.op || e.event || '·')}}</td>` +
      `<td><span class="${{statusClass(e.status, e.event)}}">${{esc(e.status || e.event || '·')}}</span></td>` +
      `<td>${{e.h === null || e.h === undefined ? '·' : `H${{e.h}} D${{e.d}}`}}</td>` +
      `<td>${{fmt(e.rq, 4)}} / ${{fmt(e.p, 3)}} / ${{fmt(e.H, 3)}}</td>` +
      `<td><code>${{esc(e.parent_short || '·')}}</code> → <code>${{esc(e.child_short || '·')}}</code></td>` +
      `<td>${{e.reservoir_saved ? esc(e.reservoir_reason || 'saved') : '·'}}</td>` +
      `<td><code>${{esc(e.previous_short || '·')}}</code> ${{e.previous_rq === null || e.previous_rq === undefined ? '' : `RQ=${{fmt(e.previous_rq, 4)}}`}}</td>` +
      `<td>${{esc(e.problem || e.error || e.reason || '')}}</td>` +
      `</tr>`).join('') +
    '</table>'
  ) : '<span class="muted">No candidate events loaded for this step.</span>';
  const programs = step.dataset.programs || [];
  document.getElementById('programTable').innerHTML = programs.length ? (
    '<table class="list"><tr><th>program_id</th><th>count</th><th>seed range</th><th>cells</th></tr>' +
    programs.map(p => `<tr><td><code>${{esc(p.program_id)}}</code></td><td>${{p.count}}</td><td>${{p.seed_min}}..${{p.seed_max}}</td><td>${{esc((p.cells || []).join(', '))}}</td></tr>`).join('') +
    '</table>'
  ) : '<span class="muted">No dataset examples for this step.</span>';
  const eventFiles = (step.events?.source_files || [])
    .filter(Boolean)
    .map(path => `<code>${{esc(path)}}</code>`)
    .join('<br>');
  document.getElementById('files').innerHTML =
    `evolution: <code>${{esc(step.source_file)}}</code><br>` +
    `dataset: <code>${{esc(step.dataset_file || 'not found')}}</code><br>` +
    `events: ${{eventFiles || '<code>not found</code>'}}`;
}}
slider.addEventListener('input', () => render(Number(slider.value), true));
eventSlider.addEventListener('input', () => render(Number(slider.value), false));
document.getElementById('play').addEventListener('click', () => {{
  let i = 0;
  const timer = setInterval(() => {{
    slider.value = i;
    render(i);
    i += 1;
    if (i >= DATA.steps.length) clearInterval(timer);
  }}, 900);
}});
document.getElementById('playEvents').addEventListener('click', () => {{
  let i = 0;
  const total = Number(eventSlider.max) || 0;
  const timer = setInterval(() => {{
    eventSlider.value = i;
    render(Number(slider.value), false);
    i += 1;
    if (i > total) clearInterval(timer);
  }}, 220);
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
{refresh_js}
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
        f"dataset_size={latest['dataset_meta'].get('dataset_size')}, "
        f"events={latest['events'].get('event_count')}"
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
        "--events_dir",
        type=Path,
        default=None,
        help="Directory containing events_evo_*_step_*.jsonl. Defaults to log_dir/events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./rq_output/evolution_map.html"),
        help="Output HTML path.",
    )
    parser.add_argument(
        "--watch_interval",
        type=float,
        default=0.0,
        help="If >0, regenerate the HTML every N seconds for live monitoring.",
    )
    parser.add_argument(
        "--auto_refresh_seconds",
        type=float,
        default=0.0,
        help="If >0, make the generated HTML reload itself every N seconds.",
    )
    parser.add_argument(
        "--n_h_bins",
        type=int,
        default=6,
        help="Fallback H-bin count when only event logs exist.",
    )
    parser.add_argument(
        "--n_d_bins",
        type=int,
        default=10,
        help="Fallback D-bin count when only event logs exist.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or (args.log_dir / "datasets")
    events_dir = args.events_dir or (args.log_dir / "events")
    auto_refresh = args.auto_refresh_seconds
    if args.watch_interval > 0 and auto_refresh <= 0:
        auto_refresh = args.watch_interval

    def _run_once() -> None:
        datasets_by_step = load_dataset_snapshots(dataset_dir)
        event_streams = load_event_streams(events_dir)
        try:
            snapshots = load_evolution_snapshots(args.log_dir)
        except FileNotFoundError:
            if not event_streams:
                raise
            snapshots = []
        payload = build_payload(
            snapshots,
            datasets_by_step,
            event_streams,
            default_shape=(args.n_h_bins, args.n_d_bins),
        )
        write_html(payload, args.output, auto_refresh)
        print_summary(payload, args.output)

    if args.watch_interval > 0:
        import time

        while True:
            _run_once()
            time.sleep(args.watch_interval)
    else:
        _run_once()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"visualize_evolution_map.py: {exc}", file=sys.stderr)
        sys.exit(1)
