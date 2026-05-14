"""
RQ-Evolve: veRL-integrated online training entry point.

Based on verl_project/main_trainer.py pattern (verl 0.3.1, OmegaConf).
R-Zero / AZR 스타일 포크 기반 접근법.

사용법:
  python run_verl.py --config configs/rq_config.yaml
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 프로젝트 내장 verl (0.3.1)을 우선 로드
# pip 설치된 verl 0.7.1보다 먼저 로드되어야 함
_PROJECT_ROOT = str(Path(__file__).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load OPENAI_API_KEY / HF_TOKEN / WANDB_API_KEY from .env (project root)
# before any module that reads os.environ. python-dotenv is a hard dep.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(_PROJECT_ROOT) / ".env", override=False)
except ImportError:
    pass

import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.config import PPOConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.tokenizer import get_tokenizer, get_processor
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.reward import BatchFunctionRewardManager
from verl.workers.reward.config import RewardConfig
from verl.trainer.data_loader import create_dataloader

from rq_questioner.program import ProblemProgram
from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.verl_dataset import MapElitesDynamicDataset
from rq_questioner.concepts import (
    concept_axis_labels,
    validate_concept_decl,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (OmegaConf, no Hydra)
# ---------------------------------------------------------------------------

def load_config(config_path: str):
    default_config = OmegaConf.structured(PPOConfig())
    user_config = OmegaConf.load(config_path) if os.path.exists(config_path) else OmegaConf.create()
    config = OmegaConf.merge(default_config, user_config)
    config = OmegaConf.to_object(config)
    config.deep_post_init()
    return config


# ---------------------------------------------------------------------------
# Seed loading + MAP-Elites init
# ---------------------------------------------------------------------------

def load_seeds(seed_dir: str) -> list[ProblemProgram]:
    programs = []
    for f in sorted(Path(seed_dir).glob("*.py")):
        prog = ProblemProgram(
            source_code=f.read_text(),
            generation=0,
            metadata={"source_file": f.name},
        )
        inst = prog.execute(seed=42)
        if inst:
            concept_type = prog.declared_concept_type()
            concept_group = prog.declared_concept_group()
            concept_reasons = validate_concept_decl(concept_type, concept_group)
            if concept_reasons:
                print(
                    f"  Seed FAIL: {f.name} "
                    f"({'; '.join(concept_reasons[:3])})"
                )
                continue
            prog.metadata["concept_type"] = concept_type
            prog.metadata["concept_group"] = concept_group
            prog.root_seed_id = prog.program_id
            programs.append(prog)
            print(f"  Seed OK: {f.name}")
        else:
            print(f"  Seed FAIL: {f.name}")
    return programs


def init_map_elites(
    seeds,
    n_h_bins,
    n_div_bins,
    h_range,
    ucb_c,
    epsilon,
    candidate_reservoir_size=4,
    diversity_axis="concept_group",
    use_reservoir=False,
) -> MAPElitesGrid:
    grid = MAPElitesGrid(
        n_h_bins=n_h_bins,
        n_div_bins=n_div_bins,
        h_range=tuple(h_range) if isinstance(h_range, list) else h_range,
        ucb_c=ucb_c,
        epsilon=epsilon,
        candidate_reservoir_size=candidate_reservoir_size,
        diversity_axis=diversity_axis,
        use_reservoir=use_reservoir,
    )

    if diversity_axis == "embedding":
        # Embedding 기반 D축: seed 문제 텍스트로 PCA fitting
        seed_problems = []
        for prog in seeds:
            for s in range(5):
                inst = prog.execute(seed=s)
                if inst:
                    seed_problems.append(inst.problem)
        if seed_problems:
            grid.fit_diversity_axis(seed_problems)
            print(f"  D-axis fitted with {len(seed_problems)} seed problems")
    else:
        print(f"  D-axis uses controlled {diversity_axis} labels")

    # Seed champion 삽입
    for prog in seeds:
        inst = prog.execute(seed=0)
        if inst:
            grid.try_insert(program=prog, h_value=1.0,
                            problem_text=inst.problem, rq_score=0.01)
    return grid


def build_seed_dataset(seeds, instances_per_program) -> MapElitesDynamicDataset:
    problems = []
    for prog in seeds:
        for seed in range(instances_per_program):
            inst = prog.execute(seed=seed)
            if inst:
                problems.append({
                    "problem": inst.problem,
                    "answer": inst.answer,
                    "program_id": prog.program_id,
                    "rq_score": 0.01,
                })
    return MapElitesDynamicDataset(seed_problems=problems)


# ---------------------------------------------------------------------------
# TaskRunner (Ray remote)
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class RQTaskRunner:
    def run(self, config):
        # Ray actors run in a fresh process — re-load .env so the trainer
        # (which runs inside this actor) sees OPENAI_API_KEY / HF_TOKEN /
        # WANDB_API_KEY without depending on Ray runtime_env propagation.
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(_PROJECT_ROOT) / ".env", override=False)
        except ImportError:
            pass

        print("=" * 60)
        print("RQ-Evolve Training")
        print("=" * 60)

        model_path = config.worker.actor.model.model_path
        trust_remote_code = getattr(config.worker.actor.model, 'trust_remote_code', True)

        print("[Runner] loading tokenizer...")
        tokenizer = get_tokenizer(model_path, trust_remote_code=trust_remote_code, use_fast=True)
        processor = get_processor(model_path, trust_remote_code=trust_remote_code, use_fast=True)
        print("[Runner] tokenizer loaded")

        # Worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }

        # Resource pool
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # Reward manager
        reward_cfg = config.worker.reward
        RemoteRewardManager = ray.remote(BatchFunctionRewardManager).options(
            num_cpus=reward_cfg.num_cpus
        )
        reward_fn = RemoteRewardManager.remote(reward_cfg, tokenizer)
        print("[Runner] reward manager created")

        # RQ config
        rq_cfg = getattr(config, 'rq', None) or {}
        if isinstance(rq_cfg, dict):
            rq_cfg_get = rq_cfg.get
        else:
            rq_cfg_get = lambda k, d=None: getattr(rq_cfg, k, d)

        seed_dir = rq_cfg_get("seed_programs_dir", "./seed_programs")
        n_h_bins = rq_cfg_get("n_h_bins", 6)
        n_div_bins = rq_cfg_get("n_div_bins", 10)
        diversity_axis = rq_cfg_get("diversity_axis", "concept_group")
        axis_labels = concept_axis_labels(diversity_axis)
        if axis_labels and n_div_bins != len(axis_labels):
            print(
                f"[Runner] overriding rq.n_div_bins={n_div_bins} to "
                f"{len(axis_labels)} for diversity_axis={diversity_axis}"
            )
            n_div_bins = len(axis_labels)
        h_range = rq_cfg_get("h_range", [0.0, 5.0])
        ucb_c = rq_cfg_get("ucb_c", 1.0)
        epsilon = rq_cfg_get("epsilon", 0.3)
        candidate_reservoir_size = rq_cfg_get("candidate_reservoir_size", 4)
        use_reservoir = bool(rq_cfg_get("use_reservoir", False))
        instances_per_program = rq_cfg_get("instances_per_program", 16)

        # Seeds + MAP-Elites
        print("[Runner] loading seed programs...")
        seeds = load_seeds(seed_dir)
        if not seeds:
            raise ValueError(f"No valid seed programs in {seed_dir}")
        print(f"[Runner] {len(seeds)} seeds loaded")

        map_elites = init_map_elites(
            seeds,
            n_h_bins,
            n_div_bins,
            h_range,
            ucb_c,
            epsilon,
            candidate_reservoir_size=candidate_reservoir_size,
            diversity_axis=diversity_axis,
            use_reservoir=use_reservoir,
        )
        print(
            f"[Runner] reservoir: "
            f"{'ON (champion + reservoir parents)' if use_reservoir else 'OFF (champion-only parents, original MAP-Elites)'}"
        )
        dynamic_dataset = build_seed_dataset(seeds, instances_per_program)
        dynamic_dataset.set_tokenizer(tokenizer, max_prompt_length=config.data.max_prompt_length)
        print(
            f"[Runner] MAP-Elites grid: {n_h_bins} x {n_div_bins} "
            f"({diversity_axis}), dataset: {len(dynamic_dataset)} problems"
        )

        # DataLoaders
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.utils.dataset import collate_fn

        train_dataloader = StatefulDataLoader(
            dataset=dynamic_dataset,
            batch_size=config.data.rollout_batch_size,
            num_workers=0,
            shuffle=config.data.shuffle,
            drop_last=True,
            collate_fn=collate_fn,
        )

        math_eval_dataloaders = {}
        if getattr(config.math_eval, "enabled", False):
            from evaluation.math_benchmarks import build_math_eval_dataloaders

            print("[Runner] loading math benchmark eval datasets...")
            math_eval_dataloaders = build_math_eval_dataloaders(
                math_eval_config=config.math_eval,
                tokenizer=tokenizer,
                max_prompt_length=config.data.max_prompt_length,
                collate_fn=collate_fn,
            )
            print(
                "[Runner] math eval benchmarks: "
                + (", ".join(sorted(math_eval_dataloaders)) or "none")
            )
        print("[Runner] dataloaders created")

        # Trainer
        from rq_questioner.verl_trainer import RQEvolveTrainer

        trainer = RQEvolveTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            # RQ-specific
            map_elites=map_elites,
            dynamic_dataset=dynamic_dataset,
            # Outer-iteration-start evolution hook (Method-aligned). Legacy
            # evolution_pct / evolution_freq are no longer consulted.
            evolve_before_train=rq_cfg_get("evolve_before_train", True),
            skip_initial_evolution_on_resume=rq_cfg_get(
                "skip_initial_evolution_on_resume", True
            ),
            inner_iterations=rq_cfg_get(
                "inner_iterations",
                rq_cfg_get("max_rounds", 8)
                * rq_cfg_get("candidates_per_evo", 8),
            ),
            inner_iteration_batch_size=rq_cfg_get(
                "inner_iteration_batch_size",
                rq_cfg_get("candidates_per_evo", 8),
            ),
            num_rollouts=rq_cfg_get("num_rollouts", 10),
            uncertainty_metric=rq_cfg_get("uncertainty_metric", "h"),
            instances_per_program=instances_per_program,
            in_depth_ratio=rq_cfg_get("in_depth_ratio", 0.5),
            crossover_ratio=rq_cfg_get("crossover_ratio", 0.2),
            # Champion re-evaluation — null (default) = all occupied champions
            # under current Solver (self-invalidating archive). int > 0 = partial
            # budget for ablation; 0 disables.
            reeval_per_step=rq_cfg_get("reeval_per_step", None),
            reeval_age_ratio=rq_cfg_get("reeval_age_ratio", 0.7),
            # Frontier band for Solver training; also tags new candidates
            # and gates reeval rebinning. Accept legacy key name as fallback.
            frontier_p_hat_range=tuple(
                rq_cfg_get(
                    "frontier_p_hat_range",
                    rq_cfg_get("reeval_evict_p_hat_range", [0.02, 0.98]),
                )
            ),
            # Training-data selection
            training_selection_mode=rq_cfg_get(
                "training_selection_mode", "h_priority_d_uniform"
            ),
            training_budget=rq_cfg_get("training_budget", None),
            strict_anti_reuse=rq_cfg_get("strict_anti_reuse", True),
            math_eval_dataloaders=math_eval_dataloaders,
        )
        print("[Runner] trainer constructed")

        trainer.init_workers()
        print("[Runner] workers initialized")

        # Initial Questioner evolution is now run by
        # trainer._pre_outer_iteration_hook() at outer iteration 0 inside
        # fit() — after checkpoint load, so resume uses the checkpointed
        # policy rather than the base model. Controlled by
        # rq.evolve_before_train and rq.skip_initial_evolution_on_resume
        # (both read during trainer construction).
        #
        # Note: Method requires max_steps to be set explicitly when evolving
        # each outer iteration, because training_steps is computed once from
        # the bootstrap dataloader and propagated to worker-side LR schedulers
        # during init_workers(). Fail loudly on max_steps=None.
        if getattr(config.trainer, "max_steps", None) is None:
            raise RuntimeError(
                "[Runner] max_steps is None. Outer-iteration evolution needs an "
                "explicit max_steps because actor/critic optim.training_steps "
                "are fixed from the bootstrap dataloader during trainer "
                "construction and cannot be updated after init_workers(). "
                "Set trainer.max_steps explicitly in the config."
            )

        trainer.fit()
        print("[Runner] training complete!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ-Evolve Training")
    parser.add_argument("--config", type=str, default="configs/rq_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # GPU check
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if config.trainer.n_gpus_per_node > available_gpus:
        print(f"Warning: requested {config.trainer.n_gpus_per_node} GPUs, only {available_gpus} available. Clamping.")
        config.trainer.n_gpus_per_node = available_gpus

    # Ray init
    if not ray.is_initialized():
        project_root = str(Path(__file__).parent.resolve())
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTHONPATH": f"{project_root}:" + os.environ.get("PYTHONPATH", ""),
        }
        # Forward secrets that the math eval / wandb / HF need into Ray
        # actors. The driver loaded these from .env at module top; Ray
        # actors don't auto-inherit os.environ, so explicit forwarding via
        # runtime_env.env_vars is required.
        for key in ("OPENAI_API_KEY", "HF_TOKEN", "WANDB_API_KEY"):
            val = os.environ.get(key)
            if val:
                env_vars[key] = val
        ray.init(address='local', runtime_env={"env_vars": env_vars}, num_cpus=32)

    print("Starting RQ-Evolve Runner...")
    runner = RQTaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
