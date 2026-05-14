"""
Evaluate a vLLM-loadable math model on the configured math benchmarks.

This script reuses evaluation/math_benchmarks.py for dataset loading,
prompt formatting, answer extraction, and grading.  It is intended for an
exported Hugging Face model directory (or HF model id) that vLLM can load.

Examples:
  python scripts/eval_vllm_math.py \
      --model /path/to/exported-hf-model \
      --config configs/rq_config_grpo.yaml \
      --output_dir rq_output/vllm_eval_grpo

  python scripts/eval_vllm_math.py \
      --model Qwen/Qwen3-4B-Base \
      --benchmark math500=test-time-compute/test_MATH:test \
      --max_samples 100
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.math_benchmarks import (  # noqa: E402
    MATH_EVAL_SYSTEM_PROMPT,
    MathBenchmarkProblem,
    grade_math_response,
    load_math_benchmark,
)


logger = logging.getLogger("eval_vllm_math")


DEFAULT_BENCHMARKS = [
    {"name": "math500", "hf_id": "test-time-compute/test_MATH", "split": "test"},
    {"name": "amc23", "hf_id": "test-time-compute/test_amc23", "split": "test"},
    {"name": "aime24", "hf_id": "test-time-compute/test_aime24", "split": "test"},
    {"name": "aime25", "hf_id": "test-time-compute/aime_2025", "split": "test"},
    {
        "name": "minerva_math",
        "hf_id": "test-time-compute/test_minerva_math",
        "split": "test",
    },
    {
        "name": "olympiadbench",
        "hf_id": "test-time-compute/test_olympiadbench",
        "split": "test",
    },
]


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def _load_config(path: str | None) -> Any:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(cfg_path)
    except ModuleNotFoundError:
        import yaml

        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def _parse_benchmark_spec(spec: str) -> dict[str, str]:
    """Parse NAME=HF_ID[:SPLIT] or NAME:HF_ID[:SPLIT]."""
    if "=" in spec:
        name, rest = spec.split("=", 1)
    elif ":" in spec:
        name, rest = spec.split(":", 1)
    else:
        name = spec
        rest = f"test-time-compute/{spec}"

    split = "test"
    # HF ids are usually owner/dataset.  Treat the final colon as split only
    # for "name=hf_id:split"; paths like C:\ are not supported on this cluster.
    if ":" in rest:
        hf_id, split = rest.rsplit(":", 1)
    else:
        hf_id = rest
    return {"name": name.strip(), "hf_id": hf_id.strip(), "split": split.strip()}


def _benchmark_specs(args: argparse.Namespace, cfg: Any) -> list[dict[str, str]]:
    if args.benchmark:
        return [_parse_benchmark_spec(spec) for spec in args.benchmark]

    specs = list(_cfg_get(cfg, "math_eval.benchmarks", []) or [])
    if not specs:
        return list(DEFAULT_BENCHMARKS)

    parsed = []
    for spec in specs:
        if isinstance(spec, str):
            parsed.append(
                {"name": spec, "hf_id": f"test-time-compute/{spec}", "split": "test"}
            )
        else:
            parsed.append(
                {
                    "name": str(spec.get("name")),
                    "hf_id": str(spec.get("hf_id")),
                    "split": str(spec.get("split", "test")),
                }
            )
    return [s for s in parsed if s["name"] and s["hf_id"]]


def _has_hf_weight_file(path: Path) -> bool:
    patterns = (
        "*.safetensors",
        "pytorch_model*.bin",
        "model*.bin",
        "model*.safetensors",
    )
    return any(next(path.glob(pattern), None) is not None for pattern in patterns)


def _looks_like_verl_fsdp_actor(path: Path) -> bool:
    return (
        path.is_dir()
        and next(path.glob("model_world_size_*_rank_*.pt"), None) is not None
        and (path / "huggingface" / "config.json").exists()
    )


def _validate_vllm_model_path(model: str) -> None:
    path = Path(model)
    if not path.exists():
        return  # HF hub id or remote path; let vLLM resolve it.
    if path.is_file():
        raise ValueError(f"--model must be a model directory or HF id, got file: {path}")
    if _has_hf_weight_file(path):
        return
    if (path / "latest_global_step.txt").exists():
        raise ValueError(
            f"{path} looks like a veRL checkpoint root, not an exported HF "
            "model. Export/merge the trained actor first, then pass the "
            "exported model directory to --model."
        )
    if _looks_like_verl_fsdp_actor(path):
        raise ValueError(
            f"{path} looks like a veRL FSDP actor checkpoint. vLLM cannot load "
            "these shards directly. Export/merge the actor to a Hugging Face "
            "model directory with weight files, then pass that directory to --model."
        )
    if (path / "config.json").exists():
        raise ValueError(
            f"{path} has config.json but no obvious HF weight files "
            "(*.safetensors or pytorch_model*.bin). Pass an exported model "
            "directory containing weights."
        )


def _build_prompt(tokenizer, problem: str, system_prompt: str) -> str:
    """R-Zero-aligned prompt builder (R-Zero/evaluation/generate.py:29-33)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            add_special_tokens=True,
        )
    # Base-model fallback (R-Zero generate.py:33): system prompt is repeated
    # once at the end of the user turn.
    return (
        f"system: {system_prompt}\n"
        f"user: {problem}\n"
        f"{system_prompt}"
    )


def _load_problems(
    specs: list[dict[str, str]],
    fast_samples: dict[str, Any],
    max_samples: int,
    sample_seed: int,
) -> dict[str, list[MathBenchmarkProblem]]:
    problems_by_benchmark = {}
    for spec in specs:
        name = spec["name"]
        if max_samples > 0:
            benchmark_max_samples = max_samples
        else:
            benchmark_max_samples = int(fast_samples.get(name, -1) or -1)
        problems = load_math_benchmark(
            name=name,
            hf_id=spec["hf_id"],
            split=spec.get("split", "test"),
            max_samples=benchmark_max_samples,
            sample_seed=sample_seed,
        )
        if problems:
            problems_by_benchmark[name] = problems
        else:
            logger.warning("Skipping %s: no valid problems loaded", name)
    return problems_by_benchmark


def _summarize(rows: list[dict[str, Any]], n: int) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "num_examples": 0,
            "pass_at_1": 0.0,
            "pass_at_n": 0.0,
            "extracted_rate": 0.0,
            "boxed_rate": 0.0,
            "avg_response_chars": 0.0,
        }
    correct_first = sum(1 for row in rows if row["correct_first"])
    correct_any = sum(1 for row in rows if row["correct_any"])
    extracted_first = sum(1 for row in rows if row["samples"][0]["extracted"])
    boxed_first = sum(1 for row in rows if row["samples"][0]["boxed"])
    response_chars = [
        len(sample["response"])
        for row in rows
        for sample in row["samples"]
    ]
    return {
        "num_examples": total,
        "n": n,
        "correct_first": correct_first,
        "correct_any": correct_any,
        "pass_at_1": correct_first / total,
        "pass_at_n": correct_any / total,
        "extracted_rate": extracted_first / total,
        "boxed_rate": boxed_first / total,
        "avg_response_chars": (
            sum(response_chars) / len(response_chars) if response_chars else 0.0
        ),
    }


def _make_sampling_params(args: argparse.Namespace, tokenizer=None):
    """R-Zero-aligned SamplingParams (generate.py:22-26): max_tokens=8192,
    temperature=0.0, stop_token_ids=[eos]."""
    from vllm import SamplingParams

    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "seed": args.seed,
    }
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["stop_token_ids"] = [int(tokenizer.eos_token_id)]
    try:
        return SamplingParams(**kwargs)
    except TypeError as exc:
        if "seed" not in str(exc):
            raise
        kwargs.pop("seed", None)
        logger.warning(
            "Installed vLLM SamplingParams does not accept seed; ignoring --seed."
        )
        return SamplingParams(**kwargs)


def evaluate(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    from vllm import LLM

    _validate_vllm_model_path(args.model)

    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    fast_samples = dict(_cfg_get(cfg, "math_eval.fast_samples", {}) or {})
    sample_seed = int(args.sample_seed)
    specs = _benchmark_specs(args, cfg)
    problems_by_benchmark = _load_problems(
        specs=specs,
        fast_samples=fast_samples,
        max_samples=int(args.max_samples),
        sample_seed=sample_seed,
    )
    if not problems_by_benchmark:
        raise RuntimeError("No benchmark problems loaded.")

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )
    sampling_params = _make_sampling_params(args, tokenizer=tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.jsonl"
    failures_path = output_dir / "failures.jsonl"

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "tokenizer": tokenizer_name,
        "config": str(args.config) if args.config else None,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.n,
            "seed": args.seed,
        },
        "benchmarks": {},
    }
    all_rows: list[dict[str, Any]] = []

    with details_path.open("w", encoding="utf-8") as details_f, failures_path.open(
        "w", encoding="utf-8"
    ) as failures_f:
        for benchmark, problems in problems_by_benchmark.items():
            logger.info("Evaluating %s (%d problems)", benchmark, len(problems))
            bench_rows: list[dict[str, Any]] = []
            prompts = [
                _build_prompt(tokenizer, item.problem, args.system_prompt)
                for item in problems
            ]
            start_time = time.time()
            # Hand the full benchmark to vLLM at once so PagedAttention /
            # continuous batching can keep the KV cache saturated across the
            # whole run. Chunking the request stream forces a tail-wait at
            # each chunk boundary and underutilises the scheduler.
            outputs = llm.generate(
                prompts,
                sampling_params,
                use_tqdm=not args.no_tqdm,
            )
            for item, output in zip(problems, outputs):
                samples = []
                for sample_idx, completion in enumerate(output.outputs):
                    response = completion.text
                    grade = grade_math_response(response, item.answer)
                    samples.append({
                        "sample_idx": sample_idx,
                        "response": response,
                        **grade,
                    })
                correct_first = bool(samples and samples[0]["correct"])
                correct_any = any(sample["correct"] for sample in samples)
                row = {
                    "benchmark": benchmark,
                    "index": item.index,
                    "problem": item.problem,
                    "answer": item.answer,
                    "metadata": item.metadata,
                    "correct_first": correct_first,
                    "correct_any": correct_any,
                    "samples": samples,
                }
                bench_rows.append(row)
                all_rows.append(row)
                details_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if not correct_first:
                    failure_row = copy.deepcopy(row)
                    if not args.log_full_failures:
                        for sample in failure_row["samples"]:
                            sample["response"] = sample["response"][
                                : args.failure_chars
                            ]
                    failures_f.write(
                        json.dumps(failure_row, ensure_ascii=False) + "\n"
                    )

            elapsed = time.time() - start_time
            bench_summary = _summarize(bench_rows, args.n)
            bench_summary["elapsed_sec"] = elapsed
            summary["benchmarks"][benchmark] = bench_summary
            logger.info(
                "%s: pass@1=%.3f pass@%d=%.3f (%d examples, %.1fs)",
                benchmark,
                bench_summary["pass_at_1"],
                args.n,
                bench_summary["pass_at_n"],
                bench_summary["num_examples"],
                elapsed,
            )

    summary["overall"] = _summarize(all_rows, args.n)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    latest_path = output_dir / "latest_summary.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== vLLM Math Eval Summary ===")
    for name, stats in summary["benchmarks"].items():
        print(
            f"{name:16s} pass@1={stats['pass_at_1']:.3f} "
            f"pass@{args.n}={stats['pass_at_n']:.3f} "
            f"n={stats['num_examples']}"
        )
    overall = summary["overall"]
    print(
        f"{'overall':16s} pass@1={overall['pass_at_1']:.3f} "
        f"pass@{args.n}={overall['pass_at_n']:.3f} "
        f"n={overall['num_examples']}"
    )
    print(f"\nSaved summary : {summary_path}")
    print(f"Saved details : {details_path}")
    print(f"Saved failures: {failures_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a vLLM-loadable model on math benchmarks."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or exported HF model directory that vLLM can load.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/id. Defaults to --model.",
    )
    parser.add_argument(
        "--config",
        default="configs/rq_config_grpo.yaml",
        help="Training config to read math_eval benchmark/sample defaults from.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help=(
            "Override config benchmarks. Format: name=hf_id[:split]. "
            "May be repeated."
        ),
    )
    parser.add_argument("--output_dir", default="rq_output/vllm_math_eval")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Global per-benchmark cap. Overrides config fast_samples if >0.",
    )
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of samples per problem. Reports pass@1 and pass@n.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", "--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--system_prompt", default=MATH_EVAL_SYSTEM_PROMPT)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--log_full_failures", action="store_true")
    parser.add_argument("--failure_chars", type=int, default=2000)
    parser.add_argument("--log_level", default="INFO")
    return parser


def _apply_config_defaults(args: argparse.Namespace, cfg: Any) -> None:
    if args.max_tokens is None:
        args.max_tokens = int(_cfg_get(cfg, "math_eval.max_tokens", 8192) or 8192)
    if args.temperature is None:
        args.temperature = float(_cfg_get(cfg, "math_eval.temperature", 0.0) or 0.0)
    if args.top_p is None:
        args.top_p = float(_cfg_get(cfg, "math_eval.top_p", 1.0) or 1.0)
    if args.sample_seed == 42:
        args.sample_seed = int(_cfg_get(cfg, "math_eval.sample_seed", 42) or 42)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    cfg = _load_config(args.config)
    _apply_config_defaults(args, cfg)
    evaluate(args, cfg)


if __name__ == "__main__":
    main()
