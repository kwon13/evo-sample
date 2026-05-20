"""Decimal-rounding re-check of an eval_vllm_math.py details.jsonl run.

Motivation
----------
math_verify sometimes rejects a boxed answer that is actually correct, just
*more precise* than the reference. E.g. a minerva_math problem whose reference
answer is "50.7" while the model boxed "50.73" -- the model is arguably more
exact, but the string/symbolic comparison fails.

This script recovers such cases deterministically (no GPT calls):

  - For every row that math_verify marked WRONG and where the model produced
    a *boxed* answer, parse both the reference answer and the model's `pred`
    as plain numbers.
  - Round the model's number to the number of decimal places the *reference*
    answer carries, then compare.
  - Only applied when the reference has >= 1 fractional digit. Integer
    references (aime, most amc) are skipped: rounding there only adds false
    positives and exact-match already handles the legitimate cases.

For math500 the `answer` field stores the full reference *solution*, so the
reference number is taken from its last \\boxed{...}.

Usage:
  python scripts/round_recheck.py \
      --details rq_output/.../details.jsonl \
      --out     rq_output/.../with_round_recheck.json [--show 30]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATH_6 = ["math500", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]

# Last \boxed{...} in a string (reference solutions may contain several).
_BOXED_RE = re.compile(r"\\boxed\s*{([^{}]+)}")
# A bare numeric literal: optional sign, digits, optional fraction, optional exp.
_NUM_RE = re.compile(r"^[+-]?\d*\.?\d+([eE][+-]?\d+)?$")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean(s: str) -> str:
    """Strip LaTeX/formatting cruft so a bare number can surface."""
    s = s.strip()
    for tok in ("$", "\\%", "%", ",", "\\!", "\\,", "\\ ", "{", "}", "\\left", "\\right"):
        s = s.replace(tok, "")
    s = s.replace("\\cdot", "").strip()
    return s.rstrip(".").strip()


def _as_number(s: str) -> float | None:
    s = _clean(s)
    if not _NUM_RE.match(s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _decimals(s: str) -> int:
    """Fractional-digit count of a cleaned numeric string, exactly as written.

    The count is the number of digits after the decimal point in the
    reference answer (digits before any 'e'); the model prediction is then
    rounded to this many places. Trailing zeros are kept ("24.310" -> 3),
    so the precision the reference is written with is respected verbatim.
    """
    s = _clean(s)
    m = re.match(r"^[+-]?\d*\.(\d+)(?:[eE].*)?$", s)
    return len(m.group(1)) if m else 0


def _ref_number(answer: str) -> tuple[float, int] | None:
    """Return (value, decimal_places) for the reference answer, or None.

    Tries the raw `answer` first; if that is not a plain number (e.g. math500
    stores a full solution), falls back to the last \\boxed{...} it contains.
    """
    candidates = [answer]
    boxed = _BOXED_RE.findall(answer)
    if boxed:
        candidates.append(boxed[-1])
    for cand in candidates:
        v = _as_number(cand)
        if v is not None:
            return v, _decimals(cand)
    return None


def _rounds_equal(pred: float, ref: float, n: int) -> bool:
    eps = 10.0 ** (-(n + 6))
    return abs(round(pred, n) - round(ref, n)) < eps


def recheck(rows: list[dict[str, Any]], require_boxed: bool) -> dict[str, Any]:
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)

    summary: dict[str, Any] = {"benchmarks": {}}
    recovered: list[dict[str, Any]] = []

    for bench in MATH_6:
        bench_rows = by_bench.get(bench, [])
        if not bench_rows:
            continue
        total = len(bench_rows)
        correct = 0
        added = 0
        skipped_intref = 0
        for row in bench_rows:
            sample = (row.get("samples") or [{}])[0]
            if sample.get("correct"):
                correct += 1
                continue
            if require_boxed and not sample.get("boxed"):
                continue
            pred_raw = sample.get("pred")
            if not pred_raw:
                continue
            ref = _ref_number(str(row.get("answer", "")))
            if ref is None:
                continue
            ref_val, ndec = ref
            if ndec < 1:
                skipped_intref += 1
                continue  # integer reference: rounding not applied
            pred_val = _as_number(str(pred_raw))
            if pred_val is None:
                continue
            if _rounds_equal(pred_val, ref_val, ndec):
                correct += 1
                added += 1
                recovered.append({
                    "benchmark": bench,
                    "index": row.get("index"),
                    "ref_answer": str(row.get("answer", ""))[:60],
                    "ref_value": ref_val,
                    "decimals": ndec,
                    "pred": str(pred_raw),
                    "problem": str(row.get("problem", ""))[:100],
                })

        pass_at_1 = correct / total if total else 0.0
        summary["benchmarks"][bench] = {
            "total": total,
            "correct": correct,
            "pass_at_1": pass_at_1,
            "rounding_recovered": added,
        }
        print(
            f"{bench:14s}: {pass_at_1 * 100:6.2f}%  ({correct}/{total})  "
            f"+{added} recovered by rounding"
        )

    if all(b in summary["benchmarks"] for b in MATH_6):
        summary["math_avg_6"] = (
            sum(summary["benchmarks"][b]["pass_at_1"] for b in MATH_6) / 6 * 100
        )
    summary["rounding_recheck"] = {
        "total_recovered": len(recovered),
        "recovered": recovered,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--details", required=True, help="details.jsonl from eval_vllm_math.py")
    p.add_argument("--out", default=None, help="output json path (default: alongside details)")
    p.add_argument("--no-require-boxed", dest="require_boxed", action="store_false",
                   help="also recheck non-boxed extracted answers (default: boxed only)")
    p.add_argument("--show", type=int, default=20, help="print up to N recovered cases")
    args = p.parse_args()

    details_path = Path(args.details).resolve()
    if not details_path.is_file():
        raise SystemExit(f"details file not found: {details_path}")

    rows = _load_jsonl(details_path)
    summary = recheck(rows, args.require_boxed)

    if "math_avg_6" in summary:
        print(f"\nMATH AVG (6, +rounding recheck): {summary['math_avg_6']:.2f}")
    rr = summary["rounding_recheck"]
    print(f"rounding recovered: {rr['total_recovered']} problem(s)")

    if args.show and rr["recovered"]:
        print(f"\n--- recovered cases (showing up to {args.show}) ---")
        for c in rr["recovered"][: args.show]:
            print(f"[{c['benchmark']} #{c['index']}] ref={c['ref_value']} "
                  f"({c['decimals']}dp)  pred={c['pred']}")

    out_path = Path(args.out) if args.out else details_path.parent / "with_round_recheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
