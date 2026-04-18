"""Download validation datasets from HuggingFace and save as parquet.

RLHFDataset expects columns:
  - `prompt` (str): problem text
  - `answer` (str): ground-truth answer

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --out_dir /workspace/evo-sample/data

Paths written match the defaults in configs/rq_config.yaml (val_files).
"""

import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = [
    {
        "hf_id": "HuggingFaceH4/MATH-500",
        "split": "test",
        "rename": {"problem": "prompt"},
        "out_name": "math500_val.parquet",
    },
]


def prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in DATASETS:
        ds = load_dataset(spec["hf_id"], split=spec["split"])
        for src, dst in spec.get("rename", {}).items():
            if src in ds.column_names:
                ds = ds.rename_column(src, dst)
        out_path = out_dir / spec["out_name"]
        ds.to_parquet(str(out_path))
        print(f"[prepare_data] {spec['hf_id']} ({spec['split']}) -> {out_path} "
              f"({len(ds)} rows, columns={ds.column_names})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory to write parquet files (default: <repo>/data).",
    )
    args = parser.parse_args()
    prepare(args.out_dir)


if __name__ == "__main__":
    main()
