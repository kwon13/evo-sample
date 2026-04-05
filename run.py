"""
R_Q Evolutionary Pipeline - Main Entry Point

Usage:
    # Step 1: Prepare seed programs from LIMR dataset
    python run.py prepare --output_dir ./seed_programs
    
    # Step 2: Run evolution (Questioner only, for debugging)
    python run.py evolve --seed_dir ./seed_programs --output_dir ./rq_output
    
    # Step 3: Run full pipeline (evolution + Solver training)
    python run.py full --seed_dir ./seed_programs --output_dir ./rq_output
"""

import argparse
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def cmd_prepare(args):
    """Prepare seed programs from LIMR dataset."""
    from data.prepare_limr import prepare_seed_programs
    prepare_seed_programs(
        output_dir=args.output_dir,
        max_static=args.max_static,
        max_limr=args.max_limr,
    )


def cmd_evolve(args):
    """Run Questioner evolution only (no Solver training)."""
    from rq_questioner.pipeline import EvolutionaryPipeline, PipelineConfig

    config = PipelineConfig(
        num_epochs=args.num_epochs,
        num_generations=args.num_generations,
        candidates_per_generation=args.candidates_per_gen,
        num_rollouts=args.num_rollouts,
        n_h_bins=args.n_h_bins,
        n_div_bins=args.n_div_bins,
        mutator_model=args.mutator_model,
        solver_model=args.solver_model,
        output_dir=args.output_dir,
        seed_programs_dir=args.seed_dir,
    )

    pipeline = EvolutionaryPipeline(config)
    pipeline.run()


def cmd_full(args):
    """Run full pipeline with veRL Solver training."""
    from rq_questioner.pipeline import EvolutionaryPipeline, PipelineConfig

    config = PipelineConfig(
        num_epochs=args.num_epochs,
        num_generations=args.num_generations,
        candidates_per_generation=args.candidates_per_gen,
        num_rollouts=args.num_rollouts,
        n_h_bins=args.n_h_bins,
        n_div_bins=args.n_div_bins,
        mutator_model=args.mutator_model,
        solver_model=args.solver_model,
        output_dir=args.output_dir,
        seed_programs_dir=args.seed_dir,
    )

    pipeline = EvolutionaryPipeline(config)
    pipeline.run()


def main():
    parser = argparse.ArgumentParser(description="R_Q Evolutionary Problem Generation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare command
    prep = subparsers.add_parser("prepare", help="Prepare seed programs from LIMR")
    prep.add_argument("--output_dir", default="./seed_programs")
    prep.add_argument("--max_static", type=int, default=10)
    prep.add_argument("--max_limr", type=int, default=500)

    # Evolve command
    evo = subparsers.add_parser("evolve", help="Run Questioner evolution only")
    evo.add_argument("--seed_dir", default="./seed_programs")
    evo.add_argument("--output_dir", default="./rq_output")
    evo.add_argument("--num_epochs", type=int, default=3)
    evo.add_argument("--num_generations", type=int, default=50)
    evo.add_argument("--candidates_per_gen", type=int, default=4)
    evo.add_argument("--num_rollouts", type=int, default=16)
    evo.add_argument("--n_h_bins", type=int, default=6)
    evo.add_argument("--n_div_bins", type=int, default=6)
    evo.add_argument("--mutator_model", default="Qwen/Qwen2.5-7B-Instruct")
    evo.add_argument("--solver_model", default="Qwen/Qwen2.5-3B")

    # Full pipeline command
    full = subparsers.add_parser("full", help="Run full pipeline")
    full.add_argument("--seed_dir", default="./seed_programs")
    full.add_argument("--output_dir", default="./rq_output")
    full.add_argument("--num_epochs", type=int, default=5)
    full.add_argument("--num_generations", type=int, default=100)
    full.add_argument("--candidates_per_gen", type=int, default=8)
    full.add_argument("--num_rollouts", type=int, default=16)
    full.add_argument("--n_h_bins", type=int, default=6)
    full.add_argument("--n_div_bins", type=int, default=6)
    full.add_argument("--mutator_model", default="Qwen/Qwen2.5-7B-Instruct")
    full.add_argument("--solver_model", default="Qwen/Qwen2.5-3B")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "evolve":
        cmd_evolve(args)
    elif args.command == "full":
        cmd_full(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
