"""
Evolutionary Pipeline: Main loop implementing Algorithm 4.1.

Outer Loop: Solver-Questioner co-evolution
  Inner Loop: Questioner evolution via MAP-Elites
    1. Sample parent programs from grid
    2. LLM mutator generates candidate programs
    3. Programs generate problem instances
    4. verify(x', a') → discard on failure
    5. H pre-filter: 1 forward pass → discard if H too low
    6. G rollouts → p_hat, R_Q
    7. Update MAP-Elites grid
  Solver Training:
    8. Select best problems from each niche
    9. Generate new instances → GRPO training
    10. Solver improves → R_Q landscape shifts → next epoch re-adapts
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .program import ProblemProgram, ProblemInstance
from .map_elites import MAPElitesGrid
from .mutator import ProgramMutator
from .verifier import verify_problem
from .rq_score import compute_rq_full, h_prefilter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the evolutionary pipeline."""
    # Outer loop
    num_epochs: int = 5
    
    # Inner loop (Questioner evolution)
    num_generations: int = 100
    candidates_per_generation: int = 8
    instances_per_program: int = 3        # k: seeds per program for evaluation
    
    # H pre-filter
    h_threshold: float = 0.1
    
    # Rollouts
    num_rollouts: int = 16                # G: rollouts for p_hat estimation
    
    # MAP-Elites
    n_h_bins: int = 6
    n_div_bins: int = 6
    h_range: tuple = (0.0, 5.0)
    
    # Mutator
    mutator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    in_depth_ratio: float = 0.7
    
    # Solver
    solver_model: str = "Qwen/Qwen2.5-3B"
    
    # Training (veRL GRPO)
    train_batch_size: int = 256
    max_gen_tokens: int = 1024
    
    # Paths
    output_dir: str = "./rq_output"
    seed_programs_dir: str = "./seed_programs"
    
    # Solver answer matching
    answer_match_method: str = "exact"    # exact, sympy, llm


class EvolutionaryPipeline:
    """
    Main pipeline implementing the R_Q-guided evolutionary problem generation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.grid = MAPElitesGrid(
            n_h_bins=config.n_h_bins,
            n_div_bins=config.n_div_bins,
            h_range=config.h_range,
        )
        self.mutator = ProgramMutator(
            model_name=config.mutator_model,
            in_depth_ratio=config.in_depth_ratio,
        )

        # Solver model (lazy loaded)
        self._solver_model = None
        self._solver_tokenizer = None

        # Logging
        self.evolution_log = []

    def load_seed_programs(self, seed_dir: str) -> list[ProblemProgram]:
        """Load initial seed programs from directory."""
        programs = []
        seed_path = Path(seed_dir)
        
        for f in sorted(seed_path.glob("*.py")):
            source = f.read_text()
            program = ProblemProgram(
                source_code=source,
                generation=0,
                metadata={"source_file": f.name},
            )
            # Smoke test
            inst = program.execute(seed=42)
            if inst is not None:
                programs.append(program)
                logger.info(f"Loaded seed program: {f.name} (id={program.program_id})")
            else:
                logger.warning(f"Seed program failed smoke test: {f.name}")

        logger.info(f"Loaded {len(programs)} seed programs")
        return programs

    def initialize_grid(self, seed_programs: list[ProblemProgram]):
        """Initialize MAP-Elites grid with seed programs."""
        # Collect sample problems for fitting diversity axis
        sample_problems = []
        for prog in seed_programs:
            for seed in range(5):
                inst = prog.execute(seed)
                if inst:
                    sample_problems.append(inst.problem)

        if sample_problems:
            self.grid.fit_diversity_axis(sample_problems)

        # Insert seed programs with default scores
        for prog in seed_programs:
            inst = prog.execute(seed=0)
            if inst:
                self.grid.try_insert(
                    program=prog,
                    h_value=1.0,  # Default until measured
                    problem_text=inst.problem,
                    rq_score=0.01,  # Small positive to populate
                )

        logger.info(f"Grid initialized: {self.grid.stats()}")

    def run(self):
        """Run the full pipeline."""
        logger.info("=" * 60)
        logger.info("Starting R_Q Evolutionary Pipeline")
        logger.info("=" * 60)

        # Load seed programs
        seed_programs = self.load_seed_programs(self.config.seed_programs_dir)
        if not seed_programs:
            raise ValueError("No valid seed programs found!")

        # Initialize grid
        self.initialize_grid(seed_programs)

        # Outer loop: Solver-Questioner co-evolution
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")

            # Inner loop: Questioner evolution
            self._evolve_questioner(epoch)

            # Solver training with evolved problems
            training_data = self._prepare_training_data(epoch)
            self._train_solver(training_data, epoch)

            # Log epoch stats
            stats = self.grid.stats()
            stats["epoch"] = epoch + 1
            self.evolution_log.append(stats)
            logger.info(f"Epoch {epoch+1} stats: {json.dumps(stats, indent=2)}")

            # Save checkpoint
            self._save_checkpoint(epoch)

        logger.info("\nPipeline complete!")
        self._save_final_results()

    def _evolve_questioner(self, epoch: int):
        """Inner loop: Evolve problem generation programs."""
        logger.info(f"[Epoch {epoch+1}] Starting Questioner evolution "
                    f"({self.config.num_generations} generations)")

        for gen in range(self.config.num_generations):
            # 1. Sample parent programs from MAP-Elites grid
            parents = []
            for _ in range(self.config.candidates_per_generation):
                parent = self.grid.sample_parent()
                if parent:
                    parents.append(parent)

            if not parents:
                logger.warning(f"  Gen {gen}: No parents available, skipping")
                continue

            # 2. LLM mutator generates candidate programs
            children = self.mutator.mutate_batch(parents)
            valid_children = [c for c in children if c is not None]

            if not valid_children:
                continue

            # 3-7. Evaluate each candidate
            inserted = 0
            for child in valid_children:
                success = self._evaluate_and_insert(child)
                if success:
                    inserted += 1

            if (gen + 1) % 10 == 0:
                logger.info(
                    f"  Gen {gen+1}/{self.config.num_generations}: "
                    f"{len(valid_children)} valid mutations, "
                    f"{inserted} inserted, "
                    f"coverage={self.grid.coverage():.2%}, "
                    f"mean_rq={self.grid.mean_rq():.4f}"
                )

    def _evaluate_and_insert(self, program: ProblemProgram) -> bool:
        """
        Evaluate a candidate program through the full pipeline:
        3. Generate instances
        4. Verify
        5. H pre-filter
        6. G rollouts → R_Q
        7. Try insert into grid
        """
        # Step 3: Generate problem instances
        seeds = list(range(self.config.instances_per_program))
        instances = program.generate_batch(seeds)
        if not instances:
            return False

        # Step 4: Verify each instance
        verified = []
        for inst in instances:
            if verify_problem(inst.problem, inst.answer):
                inst.verified = True
                verified.append(inst)

        if not verified:
            return False

        # Use the first verified instance for evaluation
        inst = verified[0]

        # Step 5: H pre-filter (measure entropy)
        h_result = self._measure_h(inst.problem)
        if h_result is None:
            return False

        h_bar = h_result
        if not h_prefilter(h_bar, self.config.h_threshold):
            return False  # H too low, skip rollouts

        # Step 6: G rollouts → p_hat, R_Q
        correct_flags = self._run_rollouts(inst)
        rq_result = compute_rq_full(correct_flags, h_bar)

        # Update program scores
        program.h_score = h_bar
        program.p_hat = rq_result.p_hat
        program.rq_score = rq_result.rq_score
        program.fitness = rq_result.rq_score

        # Step 7: Try insert into MAP-Elites grid
        return self.grid.try_insert(
            program=program,
            h_value=h_bar,
            problem_text=inst.problem,
            rq_score=rq_result.rq_score,
        )

    def _measure_h(self, problem: str) -> Optional[float]:
        """
        Measure entropy H(x') via Solver forward pass.
        
        In production, this uses the actual Solver model.
        For now, we use a simple proxy based on problem complexity.
        """
        # TODO: Replace with actual entropy measurement using Solver model
        # For now, use a heuristic proxy:
        # - Longer problems tend to have higher entropy
        # - Problems with more numbers/operations tend to be harder
        import re
        num_count = len(re.findall(r'\d+', problem))
        word_count = len(problem.split())
        
        # Rough proxy (replace with actual model-based measurement)
        h_proxy = min(5.0, 0.1 * word_count + 0.2 * num_count)
        return h_proxy

    def _run_rollouts(self, instance: ProblemInstance) -> list[bool]:
        """
        Run G rollouts of the Solver on the problem.
        
        Returns list of bool indicating correctness of each rollout.
        """
        # TODO: Replace with actual Solver rollouts
        # For now, simulate with a heuristic
        import random
        
        # Simulate: harder problems (longer) have lower pass rate
        word_count = len(instance.problem.split())
        simulated_p = max(0.05, min(0.95, 1.0 - word_count / 100.0))
        
        correct_flags = [
            random.random() < simulated_p
            for _ in range(self.config.num_rollouts)
        ]
        return correct_flags

    def _prepare_training_data(self, epoch: int) -> list[dict]:
        """
        Prepare training data from MAP-Elites grid champions.
        Each champion generates fresh instances with new seeds.
        """
        champions = self.grid.get_all_champions()
        training_data = []
        
        base_seed = epoch * 10000

        for champ in champions:
            # Generate fresh instances with new seeds
            for i in range(self.config.train_batch_size // max(1, len(champions))):
                seed = base_seed + len(training_data) + i
                inst = champ.execute(seed)
                if inst and verify_problem(inst.problem, inst.answer):
                    training_data.append({
                        "prompt": inst.problem,
                        "answer": inst.answer,
                        "program_id": champ.program_id,
                        "niche": f"{champ.niche_h}_{champ.niche_div}",
                        "rq_score": champ.rq_score,
                    })

        logger.info(f"[Epoch {epoch+1}] Prepared {len(training_data)} training instances "
                    f"from {len(champions)} champion programs")
        
        return training_data

    def _train_solver(self, training_data: list[dict], epoch: int):
        """
        Train Solver with GRPO using veRL.
        
        This prepares the dataset and launches veRL training.
        """
        if not training_data:
            logger.warning("No training data available, skipping Solver training")
            return

        # Save training data as parquet for veRL
        train_path = self.output_dir / f"train_epoch_{epoch}.parquet"
        self._save_as_parquet(training_data, train_path)

        logger.info(f"[Epoch {epoch+1}] Training data saved to {train_path}")
        logger.info(f"[Epoch {epoch+1}] Launch veRL GRPO training with:")
        logger.info(f"  python -m verl.trainer.main_ppo "
                    f"--config configs/grpo_config.yaml "
                    f"--data.train_files={train_path}")

    def _save_as_parquet(self, data: list[dict], path: Path):
        """Save training data as parquet file for veRL."""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            # veRL expects specific columns
            verl_data = []
            for _, row in df.iterrows():
                verl_data.append({
                    "data_source": "rq_evolved",
                    "prompt": [
                        {"role": "system", "content": "Solve the following math problem step by step. Put your final answer in \\boxed{}."},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": row["answer"],
                    },
                    "extra_info": {
                        "program_id": row.get("program_id", ""),
                        "niche": row.get("niche", ""),
                        "rq_score": row.get("rq_score", 0.0),
                    },
                })

            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({
                "data_source": [d["data_source"] for d in verl_data],
                "prompt": [json.dumps(d["prompt"], ensure_ascii=False) for d in verl_data],
                "ability": [d["ability"] for d in verl_data],
                "reward_model": [json.dumps(d["reward_model"], ensure_ascii=False) for d in verl_data],
                "extra_info": [json.dumps(d["extra_info"], ensure_ascii=False) for d in verl_data],
            })
            pq.write_table(table, str(path))

        except ImportError:
            # Fallback: save as jsonl
            jsonl_path = path.with_suffix(".jsonl")
            with open(jsonl_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved as JSONL (pyarrow not available): {jsonl_path}")

    def _save_checkpoint(self, epoch: int):
        """Save evolution checkpoint."""
        ckpt_dir = self.output_dir / f"checkpoint_epoch_{epoch}"
        self.grid.save(str(ckpt_dir / "grid"))

        # Save evolution log
        with open(self.output_dir / "evolution_log.json", "w") as f:
            json.dump(self.evolution_log, f, indent=2)

    def _save_final_results(self):
        """Save final results and summary."""
        summary = {
            "config": {
                k: v for k, v in vars(self.config).items()
                if not k.startswith("_")
            },
            "final_grid_stats": self.grid.stats(),
            "evolution_log": self.evolution_log,
        }
        with open(self.output_dir / "final_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save all champion programs
        champions = self.grid.get_all_champions()
        champs_dir = self.output_dir / "final_champions"
        champs_dir.mkdir(exist_ok=True)
        for champ in champions:
            champ.save(str(champs_dir / f"{champ.program_id}.json"))

        logger.info(f"Final results saved to {self.output_dir}")
        logger.info(f"Total champions: {len(champions)}")
        logger.info(f"Grid coverage: {self.grid.coverage():.2%}")
        logger.info(f"Mean R_Q: {self.grid.mean_rq():.4f}")
