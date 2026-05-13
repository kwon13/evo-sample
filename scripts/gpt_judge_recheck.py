"""GPT-4o re-check of an eval_vllm_math.py details.jsonl run.

Mirrors R-Zero's results_recheck.py:

  - Iterate every row in details.jsonl (one row per problem, with
    `samples[0].correct` set by the math_verify grader).
  - For rows where `correct_first == False` and the model emitted an
    extracted answer, ask GPT-4o whether the extracted answer is
    equivalent to the ground-truth answer.
  - Treat a `Yes` reply as a correct answer (R-Zero override).
  - Write `with_gpt_judge.json` with per-benchmark totals and the
    6-benchmark math average.

Usage:
  python scripts/gpt_judge_recheck.py \
      --details rq_output/.../details.jsonl \
      --out     rq_output/.../with_gpt_judge.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.math_benchmarks import call_gpt_judge  # noqa: E402

logger = logging.getLogger("gpt_judge_recheck")

MATH_6 = ["math500", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _first_sample(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") or []
    return samples[0] if samples else {}


def recheck(details_path: Path, gpt_model: str, api_key: str) -> dict[str, Any]:
    rows = _load_jsonl(details_path)
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)

    summary: dict[str, Any] = {"benchmarks": {}}
    calls = 0
    yes = 0
    failed = 0

    for bench in MATH_6:
        bench_rows = by_bench.get(bench, [])
        if not bench_rows:
            continue
        total = len(bench_rows)
        correct = 0
        for row in bench_rows:
            sample = _first_sample(row)
            if sample.get("correct"):
                correct += 1
                continue
            pred = sample.get("pred") or ""
            if not pred:
                continue
            res = call_gpt_judge(
                ground_truth=str(row.get("answer", "")),
                extracted_answer=str(pred),
                model=gpt_model,
                api_key=api_key,
            )
            calls += 1
            if res.error:
                failed += 1
            if res.yes:
                yes += 1
                correct += 1
            if calls % 50 == 0:
                logger.info(
                    "[%s] gpt calls so far: %d, yes: %d, failed: %d",
                    bench, calls, yes, failed,
                )
        pass_at_1 = correct / total if total else 0.0
        summary["benchmarks"][bench] = {
            "total": total,
            "correct": correct,
            "pass_at_1": pass_at_1,
        }
        print(
            f"{bench:14s}: {pass_at_1 * 100:.2f}% ({correct}/{total})  "
            f"[calls so far: {calls}, yes: {yes}]"
        )

    summary["gpt_judge"] = {"calls": calls, "yes": yes, "failed": failed}
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--details", required=True, help="path to details.jsonl from eval_vllm_math.py")
    p.add_argument("--out", required=True, help="output with_gpt_judge.json path")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set; export it (or source .env) before running."
        )

    details_path = Path(args.details).resolve()
    if not details_path.is_file():
        raise FileNotFoundError(f"details file not found: {details_path}")

    summary = recheck(details_path, args.model, api_key)

    bench_stats = summary["benchmarks"]
    if all(b in bench_stats for b in MATH_6):
        math_avg_6 = (
            sum(bench_stats[b]["pass_at_1"] for b in MATH_6) / len(MATH_6) * 100
        )
        summary["math_avg_6"] = math_avg_6
        print(f"\nMATH AVG (6, +GPT judge): {math_avg_6:.2f}")

    print(
        f"GPT judge calls: {summary['gpt_judge']['calls']}, "
        f"yes: {summary['gpt_judge']['yes']}, "
        f"failed: {summary['gpt_judge']['failed']}"
    )

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
