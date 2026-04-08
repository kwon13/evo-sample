"""
RQ-Evolve: veRL-integrated online training entry point.

모델이 단 한 번 GPU에 로드되어 evolution과 GRPO training을 공유한다.

사용법:
  python run_verl.py \
      actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
      trainer.total_epochs=5 \
      rq.evolution_freq=50 \
      rq.candidates_per_evo=8 \
      rq.num_rollouts=16

Hydra config:
  configs/rq_ppo_trainer.yaml  (veRL ppo_trainer.yaml 기반 + rq 섹션)
"""

import os
import socket
import logging

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import need_reference_policy, need_critic
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.config import validate_config
from verl.utils.import_utils import load_extern_object
from verl.experimental.reward_loop import migrate_legacy_reward_impl

from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.program import ProblemProgram
from rq_questioner.rq_score import compute_rq_full
from rq_questioner.verl_dataset import MapElitesDynamicDataset
from rq_questioner.verl_trainer import RQEvolveTrainer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
# Seed program loading & MAP-Elites init
# ---------------------------------------------------------------------------

def load_seeds(seed_dir: str) -> list[ProblemProgram]:
    from pathlib import Path
    programs = []
    for f in sorted(Path(seed_dir).glob("*.py")):
        prog = ProblemProgram(
            source_code=f.read_text(),
            generation=0,
            metadata={"source_file": f.name},
        )
        if prog.execute(seed=42):
            programs.append(prog)
            logger.info(f"Seed OK: {f.name}")
        else:
            logger.warning(f"Seed FAIL: {f.name}")
    return programs


def init_map_elites(
    seeds: list[ProblemProgram],
    n_h_bins: int,
    n_div_bins: int,
    h_range: tuple,
    ucb_c: float = 1.0,
) -> MAPElitesGrid:
    # D축 = 시드 프로그램 ID 기반
    seed_ids = [prog.program_id for prog in seeds]
    grid = MAPElitesGrid(
        n_h_bins=n_h_bins,
        n_div_bins=len(seeds),
        h_range=h_range,
        ucb_c=ucb_c,
        seed_ids=seed_ids,
    )

    # 시드 등록 + root_seed_id 설정
    for prog in seeds:
        prog.root_seed_id = prog.program_id
        grid.register_seed(prog.program_id)

    # Insert seeds with placeholder scores
    for prog in seeds:
        inst = prog.execute(seed=0)
        if inst:
            grid.try_insert(
                program=prog,
                h_value=1.0,
                problem_text=inst.problem,
                rq_score=0.01,
            )

    logger.info(f"MAP-Elites init: {grid.stats()}")
    return grid


def build_seed_dataset(
    seeds: list[ProblemProgram],
    instances_per_program: int = 3,
) -> MapElitesDynamicDataset:
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
# TaskRunner (Ray remote, mirrors main_ppo.TaskRunner)
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class RQTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer, hf_processor
        from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
        from verl.trainer.ppo.utils import Role
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        print(f"RQTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ---- Worker setup (mirrors TaskRunner.add_actor_rollout_worker) ----
        use_legacy = config.trainer.get("use_legacy_worker_impl", "auto")

        if use_legacy == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker
            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            role = Role.ActorRolloutRef if need_reference_policy(config) else Role.ActorRollout
        else:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            role = Role.ActorRollout

        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"

        if need_reference_policy(config) and use_legacy != "disable":
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # ---- Model & tokenizer ----
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_rc = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_rc)
        processor = hf_processor(local_path, trust_remote_code=trust_rc, use_fast=True)

        # ---- Resource pool ----
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

        # ---- MAP-Elites + dynamic dataset (CPU, driver side) ----
        rq_cfg = config.get("rq", {})
        seed_dir = rq_cfg.get("seed_programs_dir", "./seed_programs")
        n_h_bins = rq_cfg.get("n_h_bins", 6)
        n_div_bins = rq_cfg.get("n_div_bins", 6)
        h_range = tuple(rq_cfg.get("h_range", [0.0, 5.0]))
        instances_per_program = rq_cfg.get("instances_per_program", 3)

        ucb_c = rq_cfg.get("ucb_c", 1.0)

        seeds = load_seeds(seed_dir)
        if not seeds:
            raise ValueError(f"No valid seed programs found in {seed_dir}")

        map_elites = init_map_elites(seeds, n_h_bins, n_div_bins, h_range, ucb_c)
        dynamic_dataset = build_seed_dataset(seeds, instances_per_program)

        # ---- Validation dataset (use val_files from config, skip if null) ----
        val_dataset = None
        val_files = config.data.get("val_files", None)
        if val_files:
            from verl.trainer.main_ppo import create_rl_dataset
            val_dataset = create_rl_dataset(
                val_files,
                config.data,
                tokenizer,
                processor,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )
        train_sampler = create_rl_sampler(config.data, dynamic_dataset)

        # ---- Trainer ----
        trainer = RQEvolveTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=dynamic_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            # RQ-specific args
            map_elites=map_elites,
            dynamic_dataset=dynamic_dataset,
            evolution_freq=rq_cfg.get("evolution_freq", 50),
            candidates_per_evo=rq_cfg.get("candidates_per_evo", 8),
            num_rollouts=rq_cfg.get("num_rollouts", 16),
            instances_per_program=instances_per_program,
            in_depth_ratio=rq_cfg.get("in_depth_ratio", 0.5),
            crossover_ratio=rq_cfg.get("crossover_ratio", 0.2),
            h_threshold=rq_cfg.get("h_threshold", 0.1),
            evolution_pct=rq_cfg.get("evolution_pct", None),
            target_hard_champions=rq_cfg.get("target_hard_champions", 6),
            max_evo_attempts=rq_cfg.get("max_evo_attempts", 64),
        )
        trainer.init_workers()
        trainer.fit()


# ---------------------------------------------------------------------------
# Main entry point (Hydra)
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="configs",
    config_name="rq_ppo_trainer",
    version_base=None,
    # veRL 내부 config을 search path에 추가하기 위해
    # searchpath에 verl/trainer/config 자동 포함됨 (pkg://verl.trainer.config)
)
def main(config):
    config = migrate_legacy_reward_impl(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = RQTaskRunner.remote()
    ray.get(runner.run.remote(config))


if __name__ == "__main__":
    main()
