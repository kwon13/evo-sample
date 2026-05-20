"""Stacked re-check: GPT-4o judge THEN decimal-rounding.

For problems that are *still wrong after the GPT-4o judge*, apply the
decimal-rounding rule (reference answer must carry >= 1 fractional digit) and
report the combined pass@1 for every step under a checkpoint parent dir.

Why this is cheap
-----------------
with_gpt_judge.json only stores aggregate counts, not per-problem GPT verdicts.
But the rounding rule fires on just a handful of rows per step. So instead of
re-running the whole GPT judge, this script:

  1. Reads the bulk GPT-judge totals from each step's with_gpt_judge.json.
  2. Finds the rows the rounding rule would recover (math_verify-wrong, boxed,
     numeric reference with >= 1 decimal place, model rounds to reference).
  3. GPT-judges *only those* rows to learn whether the GPT judge already
     accepted them.
  4. final correct = gpt_judge correct + (rounding-recovered AND GPT said no).

Usage:
  python scripts/gpt_round_recheck.py <PARENT_DIR> --subdir eval
  python scripts/gpt_round_recheck.py <PARENT_DIR> --subdir eval_longlen

OPENAI_API_KEY is read from the environment or the repo .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.math_benchmarks import (  # noqa: E402
    call_gpt_judge,
    extract_math_answer,
)
from round_recheck import (  # noqa: E402
    _as_number,
    _load_jsonl,
    _ref_number,
    _rounds_equal,
)

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


def _first(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("samples") or [{}])[0]


def find_rounding_recovered(rows: list[dict[str, Any]],
                            require_boxed: bool = True) -> list[dict[str, Any]]:
    """Rows math_verify marked wrong that the decimal-rounding rule accepts."""
    out: list[dict[str, Any]] = []
    for row in rows:
        s = _first(row)
        if s.get("correct"):
            continue
        if require_boxed and not s.get("boxed"):
            continue
        pred_raw = s.get("pred")
        if not pred_raw:
            continue
        ref = _ref_number(str(row.get("answer", "")))
        if ref is None:
            continue
        ref_val, ndec = ref
        if ndec < 1:
            continue
        pred_val = _as_number(str(pred_raw))
        if pred_val is None:
            continue
        if _rounds_equal(pred_val, ref_val, ndec):
            out.append(row)
    return out


def process_step(step_dir: Path, subdir: str, api_key: str, model: str,
                  max_workers: int) -> dict[str, Any] | None:
    details = step_dir / subdir / "details.jsonl"
    gpt_json = step_dir / subdir / "with_gpt_judge.json"
    if not details.is_file() or not gpt_json.is_file():
        return None

    rows = _load_jsonl(details)
    gj = json.loads(gpt_json.read_text())
    gj_bench = gj.get("benchmarks", {})

    recovered = find_rounding_recovered(rows)

    # GPT-judge only the rounding-recovered rows: was each already accepted?
    extra: dict[str, int] = defaultdict(int)   # GPT said NO -> rounding adds it
    cases: list[dict[str, Any]] = []

    def _judge(row: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
        gt = str(row.get("answer", ""))
        # math500's `answer` is the full solution; hand GPT just the boxed
        # final answer (consistent with gpt_judge_recheck.py).
        if row.get("benchmark") == "math500":
            boxed, _ = extract_math_answer(gt)
            if boxed:
                gt = boxed
        res = call_gpt_judge(
            ground_truth=gt,
            extracted_answer=str(_first(row).get("pred", "")),
            model=model,
            api_key=api_key,
        )
        return row, bool(res.yes) and not res.error, bool(res.error)

    if recovered:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_judge, r) for r in recovered]
            for fut in as_completed(futs):
                row, gpt_yes, err = fut.result()
                b = row["benchmark"]
                if not gpt_yes:
                    extra[b] += 1
                cases.append({
                    "benchmark": b,
                    "index": row.get("index"),
                    "answer": str(row.get("answer", ""))[:50],
                    "pred": str(_first(row).get("pred", "")),
                    "gpt_judge": "yes" if gpt_yes else ("error" if err else "no"),
                    "counted_by_rounding": not gpt_yes,
                })

    # Build the stacked summary.
    out_bench: dict[str, Any] = {}
    for b in MATH_6:
        gb = gj_bench.get(b)
        if not gb:
            continue
        total = gb["total"]
        gpt_correct = gb["correct"]
        final_correct = gpt_correct + extra.get(b, 0)
        out_bench[b] = {
            "total": total,
            "gpt_correct": gpt_correct,
            "rounding_added": extra.get(b, 0),
            "correct": final_correct,
            "pass_at_1": final_correct / total if total else 0.0,
        }

    result: dict[str, Any] = {"benchmarks": out_bench}
    if all(b in out_bench for b in MATH_6):
        result["math_avg_6"] = sum(out_bench[b]["pass_at_1"] for b in MATH_6) / 6 * 100
    if "math_avg_6" in gj:
        result["math_avg_6_gpt_only"] = gj["math_avg_6"]
    result["rounding_recheck"] = {
        "candidates": len(recovered),
        "added_after_gpt": sum(extra.values()),
        "cases": cases,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parent", help="checkpoint parent dir (contains global_step_*/)")
    p.add_argument("--subdir", default="eval",
                   help="per-step eval folder (eval | eval_longlen | ...)")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--max_workers", type=int, default=16)
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

    print(f"=== GPT + rounding stacked re-check  [subdir={args.subdir}] ===\n")
    hdr = f"{'step':>5} | {'avg6 (GPT)':>11} | {'avg6 (GPT+round)':>17} | {'Δ':>6} | added"
    print(hdr)
    print("-" * len(hdr))

    rows_out = []
    for d in step_dirs:
        st = int(re.search(r"_(\d+)$", d.name).group(1))
        res = process_step(d, args.subdir, api_key, args.model, args.max_workers)
        if res is None:
            print(f"{st:>5} | (missing details.jsonl or with_gpt_judge.json)")
            continue
        gpt_avg = res.get("math_avg_6_gpt_only")
        new_avg = res.get("math_avg_6")
        added = res["rounding_recheck"]["added_after_gpt"]
        delta = (new_avg - gpt_avg) if (gpt_avg is not None and new_avg is not None) else None
        gpt_s = f"{gpt_avg:11.2f}" if gpt_avg is not None else f"{'-':>11}"
        new_s = f"{new_avg:17.2f}" if new_avg is not None else f"{'-':>17}"
        del_s = f"{delta:+6.2f}" if delta is not None else f"{'-':>6}"
        print(f"{st:>5} | {gpt_s} | {new_s} | {del_s} | {added}")
        rows_out.append((st, gpt_avg, new_avg, added, res))

        out_path = d / args.subdir / "with_gpt_round_recheck.json"
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # Per-benchmark detail of what rounding added, and the recovered cases.
    print("\n--- cases recovered by rounding AFTER GPT judge said 'no' ---")
    any_case = False
    for st, _, _, _, res in rows_out:
        for c in res["rounding_recheck"]["cases"]:
            if c["counted_by_rounding"]:
                any_case = True
                print(f"step {st:>3} [{c['benchmark']} #{c['index']}] "
                      f"ref={c['answer']}  pred={c['pred']}")
    if not any_case:
        print("(none — GPT judge already accepted every rounding-recovered case)")

    print(f"\nsaved per-step: <step>/{args.subdir}/with_gpt_round_recheck.json")


if __name__ == "__main__":
    main()
