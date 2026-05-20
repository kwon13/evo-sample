"""Re-run the GPT-4o judge for a SINGLE benchmark and merge it back into each
step's existing with_gpt_judge.json.

Use this when the judging logic changed for one benchmark (e.g. math500 now
compares against the boxed reference answer instead of the full solution) and
you do not want to re-judge the other five benchmarks.

For each <PARENT>/global_step_*/<subdir>/:
  - GPT-judge the chosen benchmark's math_verify-wrong rows (same boxed-GT
    logic as gpt_judge_recheck.py: math500 uses its \\boxed{} answer).
  - Replace benchmarks[<bench>] in with_gpt_judge.json and recompute math_avg_6.
  - Write per-problem verdicts to gpt_verdicts_<bench>.jsonl.

The global `gpt_judge` call counters in with_gpt_judge.json are left as-is
(they reflect the original full run); a `regraded` block records this pass.

Usage:
  python scripts/regrade_benchmark.py <PARENT> --subdir eval_longlen --benchmark math500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.math_benchmarks import call_gpt_judge, extract_math_answer  # noqa: E402

MATH_6 = ["math500", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]


def _load_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith("OPENAI_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _first(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("samples") or [{}])[0]


def _ground_truth(row: dict[str, Any]) -> str:
    gt = str(row.get("answer", ""))
    if row.get("benchmark") == "math500":
        boxed, _ = extract_math_answer(gt)
        if boxed:
            gt = boxed
    return gt


def regrade_step(step_dir: Path, subdir: str, bench: str, api_key: str,
                 model: str, max_workers: int) -> dict[str, Any] | None:
    details = step_dir / subdir / "details.jsonl"
    gpt_json = step_dir / subdir / "with_gpt_judge.json"
    if not details.is_file() or not gpt_json.is_file():
        return None

    rows = [r for r in _load_jsonl(details) if r.get("benchmark") == bench]
    if not rows:
        return None

    total = len(rows)
    correct = 0
    to_judge: list[dict[str, Any]] = []
    for r in rows:
        s = _first(r)
        if s.get("correct"):
            correct += 1
            continue
        if not (s.get("pred") or ""):
            continue  # no extracted answer -> stays wrong, no GPT call
        to_judge.append(r)

    verdicts: list[dict[str, Any]] = []
    calls = yes = failed = 0

    def _judge(r: dict[str, Any]):
        gt = _ground_truth(r)
        res = call_gpt_judge(
            ground_truth=gt,
            extracted_answer=str(_first(r).get("pred", "")),
            model=model,
            api_key=api_key,
        )
        return r, gt, res

    if to_judge:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(_judge, r) for r in to_judge]):
                r, gt, res = fut.result()
                calls += 1
                if res.error:
                    failed += 1
                if res.yes:
                    yes += 1
                    correct += 1
                verdicts.append({
                    "benchmark": bench,
                    "index": r.get("index"),
                    "answer": r.get("answer", ""),
                    "gpt_ground_truth": gt,
                    "pred": str(_first(r).get("pred", "")),
                    "math_verify_correct": False,
                    "gpt_yes": bool(res.yes),
                    "gpt_raw": res.raw_response,
                    "error": res.error,
                })

    pass_at_1 = correct / total if total else 0.0

    gj = json.loads(gpt_json.read_text())
    old = dict(gj.get("benchmarks", {}).get(bench, {}))
    gj.setdefault("benchmarks", {})[bench] = {
        "total": total,
        "correct": correct,
        "pass_at_1": pass_at_1,
    }
    if all(b in gj["benchmarks"] for b in MATH_6):
        gj["math_avg_6"] = sum(
            gj["benchmarks"][b]["pass_at_1"] for b in MATH_6
        ) / 6 * 100
    gj.setdefault("regraded", {})[bench] = {
        "calls": calls, "yes": yes, "failed": failed,
    }
    gpt_json.write_text(json.dumps(gj, indent=2, ensure_ascii=False))

    vp = step_dir / subdir / f"gpt_verdicts_{bench}.jsonl"
    with vp.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    return {
        "old": old, "new": gj["benchmarks"][bench],
        "avg6": gj.get("math_avg_6"),
        "calls": calls, "yes": yes, "failed": failed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parent", help="checkpoint parent dir (contains global_step_*/)")
    p.add_argument("--subdir", default="eval_longlen",
                   help="per-step eval folder (eval | eval_longlen | ...)")
    p.add_argument("--benchmark", default="math500",
                   help="benchmark(s) to re-judge, comma-separated")
    p.add_argument("--step", default="",
                   help="step number(s) to regrade, comma-separated (default: all)")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--max_workers", type=int, default=32)
    args = p.parse_args()

    parent = Path(args.parent).resolve()
    if not parent.is_dir():
        raise SystemExit(f"not a directory: {parent}")

    api_key = _load_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (env or repo .env).")

    step_dirs = sorted(
        (d for d in parent.glob("global_step_*") if d.is_dir()),
        key=lambda d: int(re.search(r"_(\d+)$", d.name).group(1)),
    )
    want_steps = {int(s) for s in args.step.replace(",", " ").split() if s.strip()}
    if want_steps:
        step_dirs = [d for d in step_dirs
                     if int(re.search(r"_(\d+)$", d.name).group(1)) in want_steps]
        if not step_dirs:
            raise SystemExit(f"no global_step_* matched --step {args.step}")

    benchmarks = [b.strip() for b in args.benchmark.replace(",", " ").split() if b.strip()]

    for bench in benchmarks:
        print(f"\n=== re-grade '{bench}'  [subdir={args.subdir}, model={args.model}] ===")
        hdr = (f"{'step':>5} | {bench+' (old)':>18} | "
               f"{bench+' (new)':>18} | {'Δ':>8} | {'avg6':>7} | gpt(y/c)")
        print(hdr)
        print("-" * len(hdr))
        for d in step_dirs:
            st = int(re.search(r"_(\d+)$", d.name).group(1))
            res = regrade_step(d, args.subdir, bench, api_key,
                               args.model, args.max_workers)
            if res is None:
                print(f"{st:>5} | (missing details.jsonl or with_gpt_judge.json)")
                continue
            old_p = res["old"].get("pass_at_1")
            new_p = res["new"]["pass_at_1"]
            old_s = f"{old_p*100:.2f}% ({res['old'].get('correct','?')})" if old_p is not None else "-"
            new_s = f"{new_p*100:.2f}% ({res['new']['correct']})"
            delta = (new_p - old_p) * 100 if old_p is not None else 0.0
            print(f"{st:>5} | {old_s:>18} | {new_s:>18} | {delta:+7.2f}pp | "
                  f"{res['avg6']:7.2f} | {res['yes']}/{res['calls']}"
                  + (f"  fail={res['failed']}" if res['failed'] else ""))

    print(f"\nupdated per step: {args.subdir}/with_gpt_judge.json + "
          f"gpt_verdicts_<bench>.jsonl")


if __name__ == "__main__":
    main()
