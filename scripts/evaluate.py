"""
Standalone evaluation script for RQ-Evolve checkpoints.

Usage:
  python scripts/evaluate.py --model_path ./rq_output/verl_ckpt/global_step_50
  python scripts/evaluate.py --model_path Qwen/Qwen3-8B-Base  # baseline
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import (
    run_evaluation,
    load_gsm8k,
    load_math500,
    load_aime2024,
    evaluate_pass_at_1,
    evaluate_mean_at_k,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on math benchmarks")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HF model path or local checkpoint directory")
    parser.add_argument("--tp", type=int, default=2,
                        help="tensor_parallel_size")
    parser.add_argument("--gpu_mem", type=float, default=0.85)
    parser.add_argument("--gsm8k_samples", type=int, default=-1,
                        help="-1 for full (1319)")
    parser.add_argument("--math500_samples", type=int, default=-1,
                        help="-1 for full (500)")
    parser.add_argument("--aime_k", type=int, default=32,
                        help="mean@k for AIME")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--output_dir", type=str, default="./rq_output/eval_results")
    args = parser.parse_args()

    # ---- Load vLLM ----
    from vllm import LLM
    logger.info(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
    )
    logger.info("Model loaded.")

    # ---- Run benchmarks ----
    results = {}

    # GSM8K — pass@1
    gsm8k = load_gsm8k(max_samples=args.gsm8k_samples)
    if gsm8k:
        logger.info(f"\n{'='*50}")
        logger.info(f"GSM8K ({len(gsm8k)} problems, pass@1)")
        logger.info(f"{'='*50}")
        results["gsm8k"] = evaluate_pass_at_1(
            llm, gsm8k, "GSM8K", max_tokens=args.max_tokens,
        )

    # MATH-500 — pass@1
    math500 = load_math500(max_samples=args.math500_samples)
    if math500:
        logger.info(f"\n{'='*50}")
        logger.info(f"MATH-500 ({len(math500)} problems, pass@1)")
        logger.info(f"{'='*50}")
        results["math500"] = evaluate_pass_at_1(
            llm, math500, "MATH-500", max_tokens=args.max_tokens,
        )

    # AIME-2024 — mean@k
    aime = load_aime2024()
    if aime:
        logger.info(f"\n{'='*50}")
        logger.info(f"AIME-2024 ({len(aime)} problems, mean@{args.aime_k})")
        logger.info(f"{'='*50}")
        results["aime2024"] = evaluate_mean_at_k(
            llm, aime, "AIME-2024", k=args.aime_k, max_tokens=args.max_tokens,
        )

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION SUMMARY — {args.model_path}")
    logger.info(f"{'='*60}")
    for name, res in results.items():
        logger.info(f"  {res.benchmark:12s}  {res.metric:8s}  "
                     f"{res.accuracy:.1%}  ({res.correct}/{res.total})")
    logger.info(f"{'='*60}")

    # ---- Save results ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = out_dir / f"eval_{model_name}_{timestamp}.json"

    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            "benchmark": res.benchmark,
            "accuracy": res.accuracy,
            "metric": res.metric,
            "correct": res.correct,
            "total": res.total,
            "details": res.details,
        }

    with open(result_path, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "timestamp": timestamp,
            "results": serializable,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
