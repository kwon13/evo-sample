# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    max_pixels: int = 4194304
    min_pixels: int = 262144
    filter_overlong_prompts: bool = True
    execution_timeout_sec: float = 5.0

    def post_init(self):
        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                self.format_prompt = None


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 0.0
    kl_target: float = 0.0
    mock_data: str = ""
    frontier_L: float = 0.3
    frontier_U: float = 0.7
    dist_lambda: float = 1.0

@dataclass
class TrainerConfig:
    # Paper notation: one outer iteration is one archive refresh followed by
    # one pass over the refreshed Solver dataset. total_epochs is kept as a
    # legacy alias for older YAMLs.
    total_outer_iterations: Optional[int] = None
    total_epochs: int = 10
    max_steps: Optional[int] = None
    project_name: str = "easy_r1"
    experiment_name: str = "demo"
    logger: Tuple[str] = ("console", "wandb")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    critic_warmup: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    val_mode: str = "reward"
    val_every_outer_iteration: Optional[bool] = None
    val_every_epoch: bool = False  # Legacy alias.
    code_validation_dataset_path: Optional[str] = None
    code_validation_batch_size: int = 8
    code_validation_workers: int = 8
    code_validation_timeout_sec: float = 2.0
    save_freq: int = -1
    save_limit: int = -1
    save_every_outer_iteration: Optional[bool] = None
    save_every_epoch: bool = True  # Legacy alias.
    save_final_checkpoint: bool = True  # Save at fit() end if not already saved on save_freq
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    default_local_dir: Optional[str] = None  # For outer-iteration snapshots and rollout logs

    def post_init(self):
        if self.total_outer_iterations is None:
            self.total_outer_iterations = self.total_epochs
        else:
            self.total_epochs = int(self.total_outer_iterations)

        if self.val_every_outer_iteration is None:
            self.val_every_outer_iteration = self.val_every_epoch
        else:
            self.val_every_epoch = bool(self.val_every_outer_iteration)

        if self.save_every_outer_iteration is None:
            self.save_every_outer_iteration = self.save_every_epoch
        else:
            self.save_every_epoch = bool(self.save_every_outer_iteration)

        # Set default_local_dir first if not provided
        if self.default_local_dir is None:
            self.default_local_dir = os.path.join("checkpoints", self.project_name, self.experiment_name)
        self.default_local_dir = os.path.abspath(self.default_local_dir)
        
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = self.default_local_dir

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)
        if self.load_checkpoint_path is not None:
            self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)


@dataclass
class AblationConfig:
    enable_questioner: bool = True
    enable_validation: bool = True
    enable_solver: bool = True

@dataclass
class RQConfig:
    seed_programs_dir: str = "./seed_programs"
    n_h_bins: int = 6
    # MAP-Elites diversity axis. Supported: concept_group, concept_type, embedding.
    diversity_axis: str = "concept_group"
    n_div_bins: int = 10
    h_range: list = field(default_factory=lambda: [0.0, 1.0])
    # R_Q uncertainty metric. Supported: h (mean token entropy), h_span_max
    # (max mean entropy over reasoning/sentence spans).
    uncertainty_metric: str = "h"
    evolution_pct: float = 0.1
    evolution_freq: int = 50
    target_hard_champions: int = 6
    # Paper notation: each inner iteration mutates/evaluates one candidate
    # program P'. The implementation batches several inner iterations for
    # throughput. max_evo_attempts, candidates_per_evo, and max_rounds are
    # legacy aliases for older YAMLs.
    inner_iterations: Optional[int] = None
    inner_iteration_batch_size: Optional[int] = None
    max_evo_attempts: int = 64
    candidates_per_evo: int = 8
    max_rounds: int = 8
    num_rollouts: int = 10
    instances_per_program: int = 16
    crossover_ratio: float = 0.2
    in_depth_ratio: float = 0.5
    ucb_c: float = 1.0
    epsilon: float = 0.3
    # Parent selection strategy (ablation toggle):
    #   "ucb"    (default) — ε-greedy + rank-based UCB selection
    #   "random"           — uniform random over material, ignoring R_Q /
    #                        UCB entirely (original MAP-Elites parent selection)
    selection_strategy: str = "ucb"
    use_reservoir: bool = False
    candidate_reservoir_size: int = 4
    # R_Q term ablation: R_Q = p(1-p) * U. Forcing a term to 1.0 removes
    # its influence so a run can isolate the other term. Both False
    # (default) = standard R_Q.
    rq_ablate_learnability: bool = False
    rq_ablate_uncertainty: bool = False
    # null = re-evaluate all occupied champions (Method-aligned default)
    # int  = partial budget (debug/ablation); 0 disables re-evaluation
    reeval_per_step: Optional[int] = None
    reeval_age_ratio: float = 0.7
    # Frontier band for Solver training: p_hat in (f_low, f_high) is the
    # learnability frontier. p_hat outside this band is still kept in the
    # archive (mutation material) but excluded from training dataset selection.
    # Also drives frontier_status tagging on new candidates.
    frontier_p_hat_range: list = field(default_factory=lambda: [0.02, 0.98])
    training_selection_mode: str = "h_priority_d_uniform"
    training_budget: Optional[int] = None
    strict_anti_reuse: bool = True
    evolve_before_train: bool = True
    skip_initial_evolution_on_resume: bool = True

    def post_init(self):
        if self.inner_iteration_batch_size is None:
            self.inner_iteration_batch_size = self.candidates_per_evo
        self.inner_iteration_batch_size = max(1, int(self.inner_iteration_batch_size))
        self.candidates_per_evo = self.inner_iteration_batch_size

        if self.inner_iterations is None:
            self.inner_iterations = int(self.max_rounds) * self.inner_iteration_batch_size
        self.inner_iterations = max(0, int(self.inner_iterations))

        # Keep legacy fields coherent for older helper scripts and logs.
        self.max_evo_attempts = self.inner_iterations
        self.max_rounds = (
            (self.inner_iterations + self.inner_iteration_batch_size - 1)
            // self.inner_iteration_batch_size
        )

@dataclass
class GPTJudgeConfig:
    """GPT-4o re-check stage (R-Zero results_recheck.py port).

    enabled=False keeps math_verify as the sole grader (training-time default).
    enabled=True turns on a second pass that calls OpenAI's chat completions
    on items math_verify scored < 0.5, dedup'd by (question, response). The
    'Yes' fraction is added back as a score promotion. Unique calls run
    through a ThreadPoolExecutor of `max_workers` size for ~10x latency win.
    """
    enabled: bool = False
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    retry_max: int = 3
    retry_backoff_seconds: list = field(default_factory=lambda: [1, 5, 30])
    max_workers: int = 8


@dataclass
class MathEvalConfig:
    enabled: bool = False
    before_train: bool = False
    every_n_outer_iterations: Optional[int] = None
    every_n_epochs: int = 1
    # R-Zero generate.py uses max_tokens=4096 with stop_token_ids=[eos].
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 128
    sample_seed: int = 42
    output_details: bool = True
    max_logged_failures: int = 20
    # R-Zero never sub-samples math benchmarks at eval time. The loader now
    # forces -1 internally; this knob is kept only for explicit debugging.
    fast_samples: dict = field(default_factory=lambda: {
        "math500": -1,
        "amc23": -1,
        "aime24": -1,
        "aime25": -1,
        "minerva_math": -1,
        "olympiadbench": -1,
    })
    # The hf_id/split fields are ignored by the new loader (sources are
    # determined by `name` to match R-Zero) but kept here to preserve the
    # YAML schema and avoid downstream config breakage.
    benchmarks: list = field(default_factory=lambda: [
        {"name": "math500"},
        {"name": "amc23"},
        {"name": "aime24"},
        {"name": "aime25"},
        {"name": "minerva_math"},
        {"name": "olympiadbench"},
    ])
    gpt_judge: GPTJudgeConfig = field(default_factory=GPTJudgeConfig)

    def post_init(self):
        if self.every_n_outer_iterations is None:
            self.every_n_outer_iterations = self.every_n_epochs
        else:
            self.every_n_epochs = int(self.every_n_outer_iterations)

@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    rq: RQConfig = field(default_factory=RQConfig)
    math_eval: MathEvalConfig = field(default_factory=MathEvalConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
