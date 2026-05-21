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
  - Write `gpt_verdicts.jsonl` next to it: one row per GPT-judged problem
    ({benchmark, index, answer, pred, gpt_yes, gpt_raw, error}), so later
    case analysis can read each verdict instead of re-calling GPT.

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
import threading
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


def recheck(
    details_path: Path,
    gpt_model: str,
    api_key: str,
    max_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _load_jsonl(details_path)
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)

    summary: dict[str, Any] = {"benchmarks": {}}
    total_calls = 0
    total_yes = 0
    total_failed = 0

    # Per-problem GPT verdicts, so case analysis does not need a GPT re-run.
    verdicts: list[dict[str, Any]] = []

    progress_lock = threading.Lock()

    def _judge_one(row: dict[str, Any]) -> tuple[dict[str, Any], str, Any]:
        gt = str(row.get("answer", ""))
        # math500 stores the full reference *solution* in `answer`; hand GPT
        # only its boxed final answer so the comparison is not diluted by the
        # surrounding derivation. Other benchmarks already store a clean answer.
        if row.get("benchmark") == "math500":
            boxed, _ = extract_math_answer(gt)
            if boxed:
                gt = boxed
        return row, gt, call_gpt_judge(
            ground_truth=gt,
            extracted_answer=str(_first_sample(row).get("pred", "")),
            model=gpt_model,
            api_key=api_key,
        )

    for bench in MATH_6:
        bench_rows = by_bench.get(bench, [])
        if not bench_rows:
            continue
        total = len(bench_rows)

        # Pre-count rows that math_verify already accepted; collect the rest
        # (incorrect + has an extracted answer) for GPT judging.
        correct = 0
        to_judge: list[dict[str, Any]] = []
        for row in bench_rows:
            sample = _first_sample(row)
            if sample.get("correct"):
                correct += 1
                continue
            pred = sample.get("pred") or ""
            if not pred:
                continue  # extraction failed; leave as wrong, no GPT call
            to_judge.append(row)

        bench_calls = 0
        bench_yes = 0
        bench_failed = 0
        if to_judge:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_judge_one, row) for row in to_judge]
                for fut in as_completed(futures):
                    row, gt_used, res = fut.result()
                    bench_calls += 1
                    if res.error:
                        bench_failed += 1
                    if res.yes:
                        bench_yes += 1
                        correct += 1
                    verdicts.append({
                        "benchmark": bench,
                        "index": row.get("index"),
                        "answer": row.get("answer", ""),
                        "gpt_ground_truth": gt_used,
                        "pred": str(_first_sample(row).get("pred", "")),
                        "math_verify_correct": False,
                        "gpt_yes": bool(res.yes),
                        "gpt_raw": res.raw_response,
                        "error": res.error,
                    })
                    if bench_calls % 50 == 0:
                        with progress_lock:
                            logger.info(
                                "[%s] gpt %d/%d  yes=%d  failed=%d",
                                bench, bench_calls, len(to_judge),
                                bench_yes, bench_failed,
                            )

        total_calls += bench_calls
        total_yes += bench_yes
        total_failed += bench_failed
        pass_at_1 = correct / total if total else 0.0
        summary["benchmarks"][bench] = {
            "total": total,
            "correct": correct,
            "pass_at_1": pass_at_1,
        }
        print(
            f"{bench:14s}: {pass_at_1 * 100:.2f}% ({correct}/{total})  "
            f"[gpt={bench_calls}, yes={bench_yes}, fail={bench_failed}]"
        )

    summary["gpt_judge"] = {
        "calls": total_calls,
        "yes": total_yes,
        "failed": total_failed,
    }
    return summary, verdicts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--details", required=True, help="path to details.jsonl from eval_vllm_math.py")
    p.add_argument("--out", required=True, help="output with_gpt_judge.json path")
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help=("ThreadPoolExecutor size for concurrent gpt-4o calls. "
              "Raise (e.g. 64) if your OpenAI tier allows; call_gpt_judge "
              "retries transient 429s with backoff."),
    )
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

    summary, verdicts = recheck(details_path, args.model, api_key, args.max_workers)

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

    # Per-problem GPT verdicts (one row per GPT-judged problem), so later case
    # analysis can read the verdict directly instead of re-calling GPT.
    verdicts_path = out_path.parent / "gpt_verdicts.jsonl"
    with verdicts_path.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"saved: {verdicts_path}  ({len(verdicts)} verdicts)")


if __name__ == "__main__":
    main()
