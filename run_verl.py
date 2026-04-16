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
        if prog.execute(seed=42):
            prog.root_seed_id = prog.program_id
            programs.append(prog)
            print(f"  Seed OK: {f.name}")
        else:
            print(f"  Seed FAIL: {f.name}")
    return programs


def init_map_elites(seeds, n_h_bins, n_div_bins, h_range, ucb_c, epsilon) -> MAPElitesGrid:
    grid = MAPElitesGrid(
        n_h_bins=n_h_bins,
        n_div_bins=n_div_bins,
        h_range=tuple(h_range) if isinstance(h_range, list) else h_range,
        ucb_c=ucb_c,
        epsilon=epsilon,
    )

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
        val_reward_fn = RemoteRewardManager.remote(reward_cfg, tokenizer)
        print("[Runner] reward managers created")

        # RQ config
        rq_cfg = getattr(config, 'rq', None) or {}
        if isinstance(rq_cfg, dict):
            rq_cfg_get = rq_cfg.get
        else:
            rq_cfg_get = lambda k, d=None: getattr(rq_cfg, k, d)

        seed_dir = rq_cfg_get("seed_programs_dir", "./seed_programs")
        n_h_bins = rq_cfg_get("n_h_bins", 6)
        n_div_bins = rq_cfg_get("n_div_bins", 10)
        h_range = rq_cfg_get("h_range", [0.0, 5.0])
        ucb_c = rq_cfg_get("ucb_c", 1.0)
        epsilon = rq_cfg_get("epsilon", 0.3)
        instances_per_program = rq_cfg_get("instances_per_program", 16)

        # Seeds + MAP-Elites
        print("[Runner] loading seed programs...")
        seeds = load_seeds(seed_dir)
        if not seeds:
            raise ValueError(f"No valid seed programs in {seed_dir}")
        print(f"[Runner] {len(seeds)} seeds loaded")

        map_elites = init_map_elites(seeds, n_h_bins, n_div_bins, h_range, ucb_c, epsilon)
        dynamic_dataset = build_seed_dataset(seeds, instances_per_program)
        dynamic_dataset.set_tokenizer(tokenizer, max_prompt_length=config.data.max_prompt_length)
        print(f"[Runner] MAP-Elites grid: {n_h_bins} x {n_div_bins}, dataset: {len(dynamic_dataset)} problems")

        # DataLoaders
        from torchdata.stateful_dataloader import StatefulDataLoader
        from verl.utils.dataset import collate_fn

        train_dataloader = StatefulDataLoader(
            dataset=dynamic_dataset,
            batch_size=config.data.rollout_batch_size,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn,
        )

        # Val dataloader
        if config.data.val_files:
            from verl.utils.dataset import RLHFDataset
            val_dataset = RLHFDataset(
                data_path=config.data.val_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.data.prompt_key,
                answer_key=config.data.answer_key,
                max_prompt_length=config.data.max_prompt_length,
            )
            val_bs = config.data.val_batch_size if config.data.val_batch_size > 0 else len(val_dataset)
            val_dataloader = StatefulDataLoader(
                dataset=val_dataset,
                batch_size=val_bs,
                num_workers=0,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            val_dataloader = StatefulDataLoader(
                dataset=dynamic_dataset,
                batch_size=min(len(dynamic_dataset), 32),
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
            )
        print("[Runner] dataloaders created")

        # Trainer
        from rq_questioner.verl_trainer import RQEvolveTrainer

        trainer = RQEvolveTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            # RQ-specific
            map_elites=map_elites,
            dynamic_dataset=dynamic_dataset,
            evolution_pct=rq_cfg_get("evolution_pct", 0.1),
            evolution_freq=rq_cfg_get("evolution_freq", 50),
            candidates_per_evo=rq_cfg_get("candidates_per_evo", 8),
            max_rounds=rq_cfg_get("max_rounds", 8),
            num_rollouts=rq_cfg_get("num_rollouts", 10),
            instances_per_program=instances_per_program,
            in_depth_ratio=rq_cfg_get("in_depth_ratio", 0.5),
            crossover_ratio=rq_cfg_get("crossover_ratio", 0.2),
            h_threshold=rq_cfg_get("h_threshold", 0.1),
            # Champion re-evaluation (option A, opt-in, 기본 비활성)
            reeval_per_step=rq_cfg_get("reeval_per_step", 0),
            reeval_age_ratio=rq_cfg_get("reeval_age_ratio", 0.7),
            reeval_evict_p_hat_range=tuple(
                rq_cfg_get("reeval_evict_p_hat_range", [0.02, 0.98])
            ),
            # Training-data selection
            training_selection_mode=rq_cfg_get(
                "training_selection_mode", "h_priority_d_uniform"
            ),
            training_budget=rq_cfg_get("training_budget", None),
            strict_anti_reuse=rq_cfg_get("strict_anti_reuse", True),
        )
        print("[Runner] trainer constructed")

        trainer.init_workers()
        print("[Runner] workers initialized")

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
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTHONPATH": f"{project_root}:" + os.environ.get("PYTHONPATH", ""),
            }
        }
        ray.init(address='local', runtime_env=runtime_env, num_cpus=32)

    print("Starting RQ-Evolve Runner...")
    runner = RQTaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
