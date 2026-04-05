"""
R_Q Self-Evolving Pipeline — Main Entry Point

One model improves at both generating problems and solving them.

Usage:
    python run.py full --model Qwen/Qwen2.5-7B-Instruct
    python run.py full --model Qwen/Qwen2.5-7B-Instruct --num_epochs 3 --num_generations 50
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def cmd_run(args):
    from rq_questioner.pipeline import EvolutionaryPipeline, PipelineConfig

    config = PipelineConfig(
        model_path=args.model,
        tp=args.tp,
        gpu_mem=args.gpu_mem,
        num_epochs=args.num_epochs,
        num_generations=args.num_generations,
        candidates_per_gen=args.candidates_per_gen,
        num_rollouts=args.num_rollouts,
        n_h_bins=args.n_h_bins,
        n_div_bins=args.n_div_bins,
        train_batch_size=args.train_batch_size,
        eval_gsm8k_samples=args.eval_gsm8k,
        eval_math500_samples=args.eval_math500,
        eval_aime_k=args.eval_aime_k,
        output_dir=args.output_dir,
        seed_programs_dir=args.seed_dir,
    )

    pipeline = EvolutionaryPipeline(config)
    pipeline.run()


def main():
    parser = argparse.ArgumentParser(description="R_Q Self-Evolving Pipeline")
    sub = parser.add_subparsers(dest="command")

    for name in ["full", "evolve"]:
        p = sub.add_parser(name)

        # Model
        p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
        p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
        p.add_argument("--gpu_mem", type=float, default=0.85)

        # Evolution
        p.add_argument("--num_epochs", type=int, default=5)
        p.add_argument("--num_generations", type=int, default=100)
        p.add_argument("--candidates_per_gen", type=int, default=8)
        p.add_argument("--num_rollouts", type=int, default=16)
        p.add_argument("--train_batch_size", type=int, default=256)

        # MAP-Elites
        p.add_argument("--n_h_bins", type=int, default=6)
        p.add_argument("--n_div_bins", type=int, default=6)

        # Evaluation
        p.add_argument("--eval_gsm8k", type=int, default=200, help="-1 for full")
        p.add_argument("--eval_math500", type=int, default=100, help="-1 for full")
        p.add_argument("--eval_aime_k", type=int, default=32)

        # Paths
        p.add_argument("--seed_dir", default="./seed_programs")
        p.add_argument("--output_dir", default="./rq_output")

    args = parser.parse_args()
    if args.command in ("full", "evolve"):
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
