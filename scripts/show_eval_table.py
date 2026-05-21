"""Print per-step math-eval results as a plain-text table (no matplotlib).

For each step under <PARENT>/global_step_*/<subdir>/, reads a result file and
prints per-benchmark pass@1 (%) plus the 6-benchmark average.

Result file by --source:
  round  -> with_gpt_round_recheck.json  (GPT judge + decimal rounding)
  gpt    -> with_gpt_judge.json          (GPT judge)
  mv     -> summary.json                 (math_verify only)
  auto   -> first of round / gpt / mv that exists  (default)

Usage:
  python scripts/show_eval_table.py <PARENT>
  python scripts/show_eval_table.py <PARENT> --subdir eval,eval_longlen
  python scripts/show_eval_table.py <PARENT> --subdir eval_longlen --source gpt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MATH_6 = ["math500", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]
SHORT = {"math500": "math500", "amc23": "amc23", "aime24": "aime24",
         "aime25": "aime25", "minerva_math": "minerva", "olympiadbench": "olympiad"}
FILES = {
    "round": "with_gpt_round_recheck.json",
    "gpt": "with_gpt_judge.json",
    "mv": "summary.json",
}


def _read(step_dir: Path, source: str) -> tuple[dict, str] | None:
    """Return ({bench: pass@1 %, 'avg6': %}, used_source) for one step."""
    order = [source] if source != "auto" else ["round", "gpt", "mv"]
    for src in order:
        f = step_dir / FILES[src]
        if not f.is_file():
            continue
        data = json.loads(f.read_text())
        bench = data.get("benchmarks", {})
        out: dict = {}
        for b in MATH_6:
            if b in bench and "pass_at_1" in bench[b]:
                out[b] = bench[b]["pass_at_1"] * 100.0
        if not out:
            continue
        if "math_avg_6" in data:
            out["avg6"] = data["math_avg_6"]
        elif all(b in out for b in MATH_6):
            out["avg6"] = sum(out[b] for b in MATH_6) / 6
        return out, src
    return None


def _step_num(d: Path) -> int:
    m = re.search(r"global_step_(\d+)$", d.name)
    return int(m.group(1)) if m else -1


def show(parent: Path, subdir: str, source: str) -> None:
    step_dirs = sorted(
        (d for d in parent.glob("global_step_*") if d.is_dir() and _step_num(d) >= 0),
        key=_step_num,
    )
    rows = []
    used = set()
    for d in step_dirs:
        r = _read(d / subdir, source)
        if r is None:
            continue
        vals, src = r
        rows.append((_step_num(d), vals))
        used.add(src)
    if not rows:
        print(f"[{subdir}] no result files found (source={source})")
        return

    src_tag = "/".join(sorted(used))
    print(f"### {parent.name}  [{subdir}]   source={src_tag}")
    head = f"{'step':>5} | " + " | ".join(f"{SHORT[b]:>8}" for b in MATH_6) + f" | {'avg6':>7}"
    print(head)
    print("-" * len(head))
    for st, v in rows:
        cells = " | ".join(
            f"{v[b]:8.2f}" if b in v else f"{'-':>8}" for b in MATH_6
        )
        avg = f"{v['avg6']:7.2f}" if "avg6" in v else f"{'-':>7}"
        print(f"{st:>5} | {cells} | {avg}")
    # column means across steps
    avgline = []
    for b in MATH_6:
        xs = [v[b] for _, v in rows if b in v]
        avgline.append(f"{sum(xs)/len(xs):8.2f}" if xs else f"{'-':>8}")
    a6 = [v["avg6"] for _, v in rows if "avg6" in v]
    print("-" * len(head))
    print(f"{'mean':>5} | " + " | ".join(avgline) +
          f" | {sum(a6)/len(a6):7.2f}" if a6 else "")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("parent", help="checkpoint parent dir (contains global_step_*/)")
    p.add_argument("--subdir", default="eval",
                   help="per-step eval folder(s), comma-separated (e.g. eval,eval_longlen)")
    p.add_argument("--source", default="auto", choices=["auto", "round", "gpt", "mv"],
                   help="which result file to read (default: auto)")
    args = p.parse_args()

    parent = Path(args.parent).resolve()
    if not parent.is_dir():
        raise SystemExit(f"not a directory: {parent}")

    for i, sub in enumerate(s.strip() for s in args.subdir.split(",") if s.strip()):
        if i:
            print()
        show(parent, sub, args.source)


if __name__ == "__main__":
    main()
