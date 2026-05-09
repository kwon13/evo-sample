"""
RQ-Evolve Trainer: veRL RayPPOTrainer + MAP-Elites evolution.

Method-aligned outer-iteration flow (method.tex §coevolution):
  매 outer iteration 시작에서 _pre_outer_iteration_hook()이 evolution을 수행.

  Outer iteration 0 .. N-1:
    0. _pre_outer_iteration_hook(outer_iteration_idx) → _outer_iteration_step():
       a. Champion re-evaluation: filled niche의 champion을 현재 Solver
          기준으로 R_Q 재측정 (self-invalidating archive)
       b. Inner iterations:
          - MAP-Elites parent 샘플링 → LLM mutation → multi-seed 자가 검증
          - G rollouts → response별 entropy 평균 → R_Q scoring
          - grid try_insert / evict
       c. _refresh_dataset(): H-priority + D-uniform + strict anti-reuse로
          outer-iteration dataset 재조립 → dataloader rebuild
    1. Solver update pass: refresh된 dataset으로 rollout → reward → REINFORCE++
    2. outer iteration 종료 시 math_eval 벤치마크 평가 (설정된 주기)

  step-based _pre_actor_update_hook trigger는 제거됨.
"""

import re
import sys
import uuid
import json
import random
import logging
import numpy as np
import torch

# 프로젝트 내장 verl (0.3.1) 우선 로드
import pathlib as _pathlib
_PROJECT_ROOT = str(_pathlib.Path(__file__).parent.parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from torchdata.stateful_dataloader import StatefulDataLoader

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.dataset import collate_fn as verl_collate_fn

from .map_elites import MAPElitesGrid
from .program import ProblemProgram, ProblemInstance
from .rq_score import compute_rq_full
from .verl_dataset import MapElitesDynamicDataset
from .code_utils import (
    extract_generator_code,
    lint_generator_source,
    lint_problem_instance,
)
from .concepts import validate_concept_decl
from evaluation.math_benchmarks import grade_math_response, save_math_eval_details

from prompts import (
    MUTATE_DEPTH, MUTATE_BREADTH, MUTATE_CROSSOVER,
    MUTATION_SYSTEM_PROMPT,
    build_score_feedback, build_few_shot_examples, build_execution_feedback,
    parent_concept_fields, choose_prefill_concept, build_mutation_prefill,
    SOLVER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = SOLVER_SYSTEM_PROMPT




_BOXED_RE = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Self-verification: multi-seed execution + SymPy answer check
# ---------------------------------------------------------------------------

def _verify_program(program: ProblemProgram, n_seeds: int = 5) -> ProblemInstance | None:
    """
    프로그램을 여러 seed로 실행해서 일관성과 답의 수학적 유효성을 검증.

    검증 기준:
      1. n_seeds개 seed 중 전부 실행 성공
      2. 모든 answer가 SymPy로 파싱 가능 (수학적으로 유효)
      3. 첫 번째 인스턴스를 반환

    Returns: 첫 인스턴스 (검증 통과) or None (실패)
    """
    from sympy import sympify

    if lint_generator_source(program.source_code):
        return None

    concept_type = program.declared_concept_type()
    concept_group = program.declared_concept_group()
    if validate_concept_decl(concept_type, concept_group):
        return None
    program.metadata["concept_type"] = concept_type
    program.metadata["concept_group"] = concept_group

    instances = []
    problems = []
    answers = []
    for s in range(n_seeds):
        inst = program.execute(seed=s, timeout=5.0)
        if inst is None:
            return None  # 하나라도 실행 실패 → 거부
        if lint_problem_instance(inst):
            return None
        # SymPy로 답이 파싱 가능한지 확인
        try:
            answer_str = inst.answer.strip().replace("^", "**")
            sympify(answer_str)
        except Exception:
            # SymPy 파싱 실패해도 float 변환 가능하면 OK
            try:
                float(inst.answer)
            except (ValueError, TypeError):
                return None  # 수학적으로 무효한 답
        instances.append(inst)
        problems.append(_normalize(inst.problem))
        answers.append(_normalize(inst.answer))

    if n_seeds > 1 and len(set(zip(problems, answers))) <= 1:
        return None
    return instances[0] if instances else None






def _extract_boxed(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def _normalize(s: str) -> str:
    s = s.strip().lower()
    for o, c in [("{", "}"), ("[", "]"), ("(", ")")]:
        if s.startswith(o) and s.endswith(c):
            s = s[1:-1].strip()
    return " ".join(s.rstrip(".").replace(",", "").split())


def _sympy_equal(a: str, b: str, tol: float = 1e-4) -> bool | None:
    """SymPy로 두 수학 표현식의 동치 판별."""
    from sympy import sympify, N, simplify
    from sympy.parsing.latex import parse_latex

    def _parse(s: str):
        s = s.strip().replace("^", "**")
        if "\\" in s:
            try:
                return parse_latex(s)
            except Exception:
                pass
        try:
            return sympify(s)
        except Exception:
            pass
        return None

    expr_a, expr_b = _parse(a), _parse(b)
    if expr_a is None or expr_b is None:
        return None
    try:
        if simplify(expr_a - expr_b) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(float(N(expr_a)) - float(N(expr_b))) < tol
    except Exception:
        pass
    return None


def _answers_match(pred: str, gt: str) -> bool:
    if _normalize(pred) == _normalize(gt):
        return True
    result = _sympy_equal(pred, gt)
    if result is not None:
        return result
    return False


def _canonical_uncertainty_metric(metric: str) -> str:
    return metric


def _select_uncertainty_score(
    metric: str,
    h_bar: float,
    flags: list[bool],
    pred_answers: list[str | None],
    h_span_max: float | None = None,
) -> tuple[float, dict[str, float]]:
    metric = _canonical_uncertainty_metric(metric)
    span_u = h_bar if h_span_max is None else h_span_max
    scores = {
        "h": float(h_bar),
        "h_span_max": float(span_u),
    }
    if metric not in scores:
        raise ValueError(f"Unknown uncertainty_metric: {metric}")
    return scores[metric], scores


def _extract_code(text: str) -> str | None:
    return extract_generator_code(text)


def _format_mutation_prompt(tokenizer, prompt_text: str, suffix: str) -> str:
    """Wrap a mutation prompt in chat format while keeping the assistant prefill.

    Layout:
      - system turn: MUTATION_SYSTEM_PROMPT (induces internal deliberation,
        suppresses visible chain-of-thought; the prefill restricts visible
        output to code, this guides the unseen forward-pass planning)
      - user turn: instructions, parent code, rubric (= prompt_text)
      - assistant turn (partial, not closed): code-fence + prefill body (= suffix)

    Uses `continue_final_message=True` so the assistant turn stays open and the
    model continues directly from the prefill. If the tokenizer rejects either
    the system role or the option, falls back to raw concatenation with the
    system prompt prepended so the guidance is preserved.
    """
    fallback = MUTATION_SYSTEM_PROMPT + "\n\n" + prompt_text + suffix
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return fallback

    messages = [
        {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": suffix.lstrip("\n")},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except (TypeError, ValueError):
        return fallback


def _make_gen_batch(
    tokenizer,
    prompts_text: list[str],
    answers: list[str],
    temperature: float,
    eos_token_id: int,
    pad_token_id: int,
    max_prompt_length: int = 1024,
    n_repeat: int = 1,
) -> DataProto:
    """
    verl 0.3.1 generate_sequences용 DataProto 생성.
    tokenize하여 input_ids, attention_mask, position_ids, raw_prompt_ids 포함.
    """
    from tensordict import TensorDict
    from verl.utils import torch_functional as VF

    all_input_ids = []
    all_attention_mask = []
    all_position_ids = []
    all_raw_prompt_ids = []

    for prompt in prompts_text:
        encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation="left",
        )
        raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_raw_prompt_ids.append(raw_prompt_ids)

    B = len(prompts_text)

    # n_repeat > 1: non_tensor_batch도 repeat (vllm rollout이 tensor만 repeat하므로)
    if n_repeat > 1:
        rep_raw_ids = []
        rep_answers = []
        for i in range(B):
            for _ in range(n_repeat):
                rep_raw_ids.append(all_raw_prompt_ids[i])
                rep_answers.append(answers[i])
        input_ids_t = torch.stack(all_input_ids).repeat_interleave(n_repeat, dim=0)
        attn_mask_t = torch.stack(all_attention_mask).repeat_interleave(n_repeat, dim=0)
        pos_ids_t = torch.stack(all_position_ids).repeat_interleave(n_repeat, dim=0)
        actual_B = B * n_repeat
    else:
        rep_raw_ids = all_raw_prompt_ids
        rep_answers = answers
        input_ids_t = torch.stack(all_input_ids)
        attn_mask_t = torch.stack(all_attention_mask)
        pos_ids_t = torch.stack(all_position_ids)
        actual_B = B

    batch = DataProto(
        batch=TensorDict({
            "input_ids": input_ids_t,
            "attention_mask": attn_mask_t,
            "position_ids": pos_ids_t,
        }, batch_size=[actual_B]),
        non_tensor_batch={
            "raw_prompt_ids": np.array(rep_raw_ids, dtype=object),
            "ground_truth": np.array(rep_answers, dtype=object),
        },
    )
    batch.meta_info = {
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "temperature": temperature,
        "do_sample": True,
        "n": 1,  # 이미 repeat했으므로 vllm은 n=1
    }
    return batch


# ---------------------------------------------------------------------------
# RQEvolveTrainer
# ---------------------------------------------------------------------------

class RQEvolveTrainer(RayPPOTrainer):
    """
    RayPPOTrainer + MAP-Elites evolution (Method-aligned outer-iteration flow).

    fit() 흐름:
      outer iteration 0 .. N-1:
        1. _pre_outer_iteration_hook(outer_iteration_idx) → _outer_iteration_step():
             a. Champion re-evaluation (self-invalidating archive)
             b. Inner iterations (G rollouts → entropy mean → scoring)
             c. _refresh_dataset() → dataloader rebuild
        2. 이 outer iteration 의 batch 들을 순회 → Solver rollout → REINFORCE++ update
        3. outer iteration 종료 시 math_eval 벤치마크 평가

    로깅:
      - evo/* : grid 상태, champion 분포, accept rate, reeval stats (Tracker 경유)
      - math_eval/* : external benchmark pass@1 and extraction stats
      - evolution_logs/*.json : grid 스냅샷 (파일)
    """

    def __init__(
        self,
        *args,
        map_elites: MAPElitesGrid,
        dynamic_dataset: MapElitesDynamicDataset,
        inner_iterations: int | None = None,
        inner_iteration_batch_size: int | None = None,
        # Legacy aliases accepted for older call sites/configs.
        candidates_per_evo: int | None = None,
        max_rounds: int | None = None,
        num_rollouts: int = 16,
        uncertainty_metric: str = "h",
        instances_per_program: int = 3,
        in_depth_ratio: float = 0.5,
        crossover_ratio: float = 0.2,
        # --- Champion re-evaluation (Method-aligned self-invalidating archive) ---
        # None (default) — re-evaluate every occupied champion each outer iteration.
        # int > 0        — partial budget (debug/ablation, age-weighted sampling).
        # 0              — disabled (ablation only; stale R_Q allowed in archive).
        reeval_per_step: int | None = None,
        reeval_age_ratio: float = 0.7,
        # Frontier band — candidates with p_hat in this open interval are
        # "learnability frontier" and eligible for Solver training. Outside
        # the band they stay in the archive as mutation material but are
        # excluded from training dataset selection. Also drives
        # frontier_status tags and gated reeval rebinning.
        frontier_p_hat_range: tuple[float, float] = (0.02, 0.98),
        # --- Training-data selection (H-priority + D-uniform + strict anti-reuse) ---
        training_selection_mode: str = "h_priority_d_uniform",
        training_budget: int | None = None,
        strict_anti_reuse: bool = True,
        # --- Outer-iteration-start evolution hook ---
        evolve_before_train: bool = True,
        skip_initial_evolution_on_resume: bool = True,
        math_eval_dataloaders: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.map_elites = map_elites
        self.dynamic_dataset = dynamic_dataset
        if inner_iteration_batch_size is None:
            inner_iteration_batch_size = (
                candidates_per_evo if candidates_per_evo is not None else 8
            )
        inner_iteration_batch_size = max(1, int(inner_iteration_batch_size))
        if inner_iterations is None:
            if max_rounds is not None:
                inner_iterations = int(max_rounds) * inner_iteration_batch_size
            else:
                inner_iterations = 64
        inner_iterations = max(0, int(inner_iterations))

        self.inner_iterations = inner_iterations
        self.inner_iteration_batch_size = inner_iteration_batch_size
        # Legacy attributes retained for old dashboards/scripts.
        self.candidates_per_evo = inner_iteration_batch_size
        self.max_rounds = (
            (inner_iterations + inner_iteration_batch_size - 1)
            // inner_iteration_batch_size
        )
        self.num_rollouts = num_rollouts
        uncertainty_metric = _canonical_uncertainty_metric(uncertainty_metric)
        allowed_uncertainty = {
            "h", "h_span_max",
        }
        if uncertainty_metric not in allowed_uncertainty:
            raise ValueError(f"Unknown uncertainty_metric: {uncertainty_metric}")
        self.uncertainty_metric = uncertainty_metric
        self.instances_per_program = instances_per_program
        self.in_depth_ratio = in_depth_ratio
        self.crossover_ratio = crossover_ratio
        # re-evaluation
        self.reeval_per_step = reeval_per_step
        self.reeval_age_ratio = max(0.0, min(1.0, reeval_age_ratio))
        self.frontier_p_hat_range = tuple(frontier_p_hat_range)
        # training-data selection state
        self.training_selection_mode = training_selection_mode
        if training_budget is None:
            training_budget = (
                map_elites.n_h_bins * map_elites.n_div_bins * instances_per_program
            )
        self.training_budget = int(training_budget)
        self.strict_anti_reuse = strict_anti_reuse
        # outer-iteration hook flags (resume detection uses load_checkpoint_path presence)
        self.evolve_before_train = bool(evolve_before_train)
        self.skip_initial_evolution_on_resume = bool(skip_initial_evolution_on_resume)
        self._is_resume = bool(
            getattr(self.config.trainer, "load_checkpoint_path", None)
        )
        self.used_seeds: dict[str, set[int]] = {}
        self._evolution_event_seq = 0
        self._current_event_path: str | None = None
        self._latest_event_path: str | None = None
        self.math_eval_dataloaders = math_eval_dataloaders or {}

    # ------------------------------------------------------------------
    # Hook: called per mini-batch update
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # DP-safe worker invocation — pads odd batches to world_size then unpads.
    # evolution/reeval produce arbitrary batch sizes (n_children, G × candidates)
    # that may be odd under 2-GPU DP, tripping
    # DataProto.chunk(world_size) assertions. Same padding logic is used by math_eval.
    # ------------------------------------------------------------------

    def _generate_sequences_dp_safe(self, batch: DataProto) -> DataProto:
        world_size = self.actor_rollout_wg.world_size
        batch, pad = pad_dataproto_to_divisor(batch, world_size)
        out = self.actor_rollout_wg.generate_sequences(batch)
        return unpad_dataproto(out, pad)

    def _compute_log_probs_dp_safe(
        self, batch: DataProto, calculate_entropy: bool, temperature: float,
    ) -> DataProto:
        # compute_log_prob does two levels of splitting:
        #   1. DP dispatch: chunk(world_size)
        #   2. Actor micro-batch: split(micro_batch_size_per_device_for_experience)
        # Per-rank length must therefore be a non-zero multiple of micro_batch_size,
        # so the overall padded length needs to be a multiple of
        # world_size × micro_batch_size. Otherwise split() divides by zero when
        # the batch is tiny (reeval with 1–2 champions, evolution with
        # G × candidates).
        world_size = self.actor_rollout_wg.world_size
        try:
            mbs = int(self.config.worker.actor.micro_batch_size_per_device_for_experience)
            mbs = max(1, mbs)
        except AttributeError:
            mbs = 1
        divisor = world_size * mbs
        batch, pad = pad_dataproto_to_divisor(batch, divisor)
        batch.meta_info["calculate_entropy"] = calculate_entropy
        batch.meta_info["temperature"] = temperature
        out = self.actor_rollout_wg.compute_log_probs(batch)
        return unpad_dataproto(out, pad)

    def _pre_outer_iteration_hook(self, outer_iteration_idx: int) -> dict | None:
        """
        Method-aligned outer-iteration-start hook.

        fit()의 매 outer iteration 시작에서 호출되어 Questioner evolution 을
        돌린다. _outer_iteration_step() 내부에서 champion re-evaluation →
        inner iterations → dataset refresh 가 일어나므로, 해당 outer
        iteration 의 첫 batch 는 refresh 된 archive-selected dataset 을
        사용한다.

        Outer iteration 0 에서만 추가 분기:
          - evolve_before_train=false → skip
          - resume + skip_initial_evolution_on_resume=true → skip
            (checkpoint 는 이미 fit() 시작부에서 load 되었으므로, 원한다면
             resume 에서도 evolution 을 돌리는 것이 safe — skip 은 옵션)
        """
        if outer_iteration_idx == 0:
            if not self.evolve_before_train:
                logger.info(
                    "[Evolution] outer iteration 0: skipped "
                    "(evolve_before_train=false)"
                )
                return None
            if self._is_resume and self.skip_initial_evolution_on_resume:
                logger.info(
                    "[Evolution] outer iteration 0: skipped "
                    "(resume + skip_initial_evolution_on_resume=true)"
                )
                return None

        logger.info(f"[Evolution] Outer iteration {outer_iteration_idx} start")
        evo_metrics = self._outer_iteration_step()

        # 핵심 지표 요약 출력 (console에서 한눈에 파악)
        print(
            f"\n{'='*60}\n"
            f"[Evolution Summary] outer_iteration={outer_iteration_idx}\n"
            f"  Reeval:   {evo_metrics['reeval_count']} updated, "
            f"{evo_metrics['reeval_evicted']} evicted "
            f"(exec_fail={evo_metrics.get('reeval_exec_fail_evicted', 0)}), "
            f"{evo_metrics['reeval_bin_shifted']} bin-shifted "
            f"(ΔR_Q={evo_metrics['reeval_rq_delta_mean']:+.4f})\n"
            f"  Mutation: {evo_metrics['attempted']} attempted, "
            f"{evo_metrics['inserted']} inserted "
            f"({evo_metrics['accept_rate']:.0%} accept)\n"
            f"  Inner:    {evo_metrics['inner_iterations']} iterations "
            f"in {evo_metrics['inner_iteration_batches']} batches "
            f"(batch_size={evo_metrics['inner_iteration_batch_size']})\n"
            f"  Grid: {evo_metrics['grid_champions']}/{evo_metrics['total_niches']} niches filled "
            f"({evo_metrics['grid_coverage']:.0%} coverage), "
            f"H2+={evo_metrics['hard_champions']}, "
            f"reservoir={evo_metrics.get('reservoir_candidates', 0)}\n"
            f"  R_Q:  mean={evo_metrics['grid_mean_rq']:.4f}, "
            f"max={evo_metrics['grid_max_rq']:.4f}\n"
            f"  Champions: p_hat={evo_metrics['champion_p_hat_mean']:.2f}±"
            f"{evo_metrics['champion_p_hat_std']:.2f}, "
            f"H={evo_metrics['champion_h_mean']:.2f}±"
            f"{evo_metrics['champion_h_std']:.2f}\n"
            f"  Dataset: {evo_metrics['dataset_size']} problems\n"
            f"{'='*60}"
        )

        return evo_metrics

    def _pre_epoch_hook(self, epoch_idx: int) -> dict | None:
        """Legacy hook name retained for older trainer loops."""
        return self._pre_outer_iteration_hook(epoch_idx)

    def _pre_train_hook(self) -> dict | None:
        """Optionally run math benchmark eval before any outer iteration/evolution."""
        cfg = getattr(self.config, "math_eval", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if not getattr(cfg, "before_train", False):
            return None
        if not self.math_eval_dataloaders:
            logger.warning("[MathEval] before_train requested but no benchmark dataloaders are available")
            return None
        return self._validate_math_benchmarks(
            outer_iteration_idx=-1, phase="before_train"
        )

    def _post_outer_iteration_hook(self, outer_iteration_idx: int) -> dict | None:
        """Run external math benchmark eval after one outer iteration."""
        cfg = getattr(self.config, "math_eval", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        if not self.math_eval_dataloaders:
            logger.warning("[MathEval] enabled but no benchmark dataloaders are available")
            return None
        every = int(
            getattr(
                cfg,
                "every_n_outer_iterations",
                getattr(cfg, "every_n_epochs", 1),
            )
            or 1
        )
        if every <= 0:
            return None
        outer_iteration_no = outer_iteration_idx + 1
        if outer_iteration_no % every != 0:
            return None
        return self._validate_math_benchmarks(
            outer_iteration_idx, phase="outer_iteration"
        )

    def _post_epoch_hook(self, epoch_idx: int) -> dict | None:
        """Legacy hook name retained for older trainer loops."""
        return self._post_outer_iteration_hook(epoch_idx)

    def _validate_math_benchmarks(
        self, outer_iteration_idx: int, phase: str = "outer_iteration",
    ) -> dict:
        """Evaluate current policy on configured math benchmarks via veRL/vLLM."""
        cfg = self.config.math_eval
        outer_iteration_no = (
            outer_iteration_idx + 1 if outer_iteration_idx >= 0 else 0
        )
        metrics: dict[str, float] = {}
        details_payload = {
            "outer_iteration": outer_iteration_idx,
            "outer_iteration_no": outer_iteration_no,
            # Legacy fields retained for existing analysis scripts.
            "epoch": outer_iteration_idx,
            "epoch_no": outer_iteration_no,
            "phase": phase,
            "global_step": int(getattr(self, "global_step", 0) or 0),
            "benchmarks": {},
        }
        accuracies: list[float] = []
        competition_accs: list[float] = []
        competition = {"amc23", "aime24", "aime25", "olympiadbench"}

        logger.info(
            "[MathEval] phase=%s outer_iteration=%s step=%s benchmarks=%s",
            phase,
            outer_iteration_no,
            getattr(self, "global_step", "?"),
            ",".join(sorted(self.math_eval_dataloaders)),
        )

        for name, dataloader in self.math_eval_dataloaders.items():
            total = correct = extracted = boxed = 0
            failures = []
            for batch_dict in dataloader:
                test_batch = DataProto.from_single_dict(batch_dict)
                if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                    gen_batch = test_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = test_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                gen_batch.meta_info = {}
                gen_batch.meta_info.update({
                    "n": 1,
                    "temperature": float(getattr(cfg, "temperature", 0.0)),
                    "top_p": float(getattr(cfg, "top_p", 1.0)),
                    "max_tokens": int(getattr(cfg, "max_tokens", self.config.data.max_response_length)),
                    "min_pixels": self.config.data.min_pixels,
                    "max_pixels": self.config.data.max_pixels,
                })
                gen_batch, pad_size = pad_dataproto_to_divisor(
                    gen_batch, self.actor_rollout_wg.world_size
                )
                out_batch = self.actor_rollout_wg.generate_sequences(gen_batch)
                out_batch = unpad_dataproto(out_batch, pad_size=pad_size)

                response_ids = out_batch.batch["responses"]
                response_mask = out_batch.batch["response_mask"].bool()
                ground_truths = test_batch.non_tensor_batch["ground_truth"].tolist()
                problems = test_batch.non_tensor_batch.get("problem", np.array([""] * len(ground_truths), dtype=object)).tolist()
                indices = test_batch.non_tensor_batch.get("problem_index", np.arange(len(ground_truths))).tolist()

                for i, gt in enumerate(ground_truths):
                    valid_ids = response_ids[i][response_mask[i]]
                    response = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                    grade = grade_math_response(response, str(gt))
                    total += 1
                    correct += int(bool(grade["correct"]))
                    extracted += int(bool(grade["extracted"]))
                    boxed += int(bool(grade["boxed"]))
                    if (
                        not grade["correct"]
                        and len(failures) < int(getattr(cfg, "max_logged_failures", 20))
                    ):
                        failures.append({
                            "index": int(indices[i]) if str(indices[i]).isdigit() else str(indices[i]),
                            "problem": str(problems[i])[:1200],
                            "ground_truth": str(gt),
                            "prediction": grade["pred"],
                            "extract_method": grade["extract_method"],
                            "response": response[:2000],
                        })

            acc = correct / total if total else 0.0
            extract_rate = extracted / total if total else 0.0
            boxed_rate = boxed / total if total else 0.0
            accuracies.append(acc)
            if name in competition:
                competition_accs.append(acc)
            metrics[f"math_eval/{name}/pass1"] = acc
            metrics[f"math_eval/{name}/extract_rate"] = extract_rate
            metrics[f"math_eval/{name}/boxed_rate"] = boxed_rate
            metrics[f"math_eval/{name}/total"] = float(total)
            metrics[f"math_eval/{name}/correct"] = float(correct)
            details_payload["benchmarks"][name] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "extract_rate": extract_rate,
                "boxed_rate": boxed_rate,
                "failures": failures,
            }
            logger.info(
                "[MathEval] %-15s pass@1=%6.2f%% (%d/%d), extract=%5.1f%% boxed=%5.1f%%",
                name,
                100.0 * acc,
                correct,
                total,
                100.0 * extract_rate,
                100.0 * boxed_rate,
            )

        metrics["math_eval/avg_all"] = float(np.mean(accuracies)) if accuracies else 0.0
        metrics["math_eval/avg_competition"] = (
            float(np.mean(competition_accs)) if competition_accs else 0.0
        )
        metrics["math_eval/before_train"] = 1.0 if phase == "before_train" else 0.0
        if getattr(cfg, "output_details", True):
            path = save_math_eval_details(
                self.config.trainer.default_local_dir,
                global_step=int(getattr(self, "global_step", 0) or 0),
                payload=details_payload,
                outer_iteration=outer_iteration_no,
                epoch=outer_iteration_no,
            )
            metrics["math_eval/details_written"] = 1.0
            logger.info("[MathEval] details saved: %s", path)
        return metrics

    # ------------------------------------------------------------------
    # Evolution step (driver-side, CPU)
    # ------------------------------------------------------------------

    def _start_evolution_event_log(self) -> None:
        """Open a per-outer-iteration JSONL event stream for live MAP visualization."""
        import os

        self._evolution_event_seq += 1
        step = getattr(self, "global_step", 0)
        event_dir = os.path.join(
            getattr(self.config.trainer, 'default_local_dir', './rq_output'),
            "evolution_logs",
            "events",
        )
        os.makedirs(event_dir, exist_ok=True)
        self._current_event_path = os.path.join(
            event_dir, f"events_evo_{self._evolution_event_seq}_step_{step}.jsonl"
        )
        self._latest_event_path = os.path.join(event_dir, "latest_events.jsonl")
        header = {
            "event": "outer_iteration_start",
            "event_seq": 0,
            "outer_iteration_index": self._evolution_event_seq,
            # Legacy key for existing visualization scripts.
            "evolution_index": self._evolution_event_seq,
            "global_step": step,
            "inner_iterations": self.inner_iterations,
            "inner_iteration_batch_size": self.inner_iteration_batch_size,
            # Legacy keys for existing visualization scripts.
            "max_rounds": self.max_rounds,
            "candidates_per_evo": self.candidates_per_evo,
        }
        for path in (self._current_event_path, self._latest_event_path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")

    def _record_evolution_event(self, event: dict) -> None:
        """Append one candidate-level event to the current evolution JSONL log."""
        if not self._current_event_path:
            return
        import time

        event = dict(event)
        event.setdefault("global_step", getattr(self, "global_step", 0))
        event.setdefault("outer_iteration_index", self._evolution_event_seq)
        event.setdefault("evolution_index", self._evolution_event_seq)
        event["event_seq"] = getattr(self, "_last_event_seq", 0) + 1
        self._last_event_seq = event["event_seq"]
        event["time"] = time.time()
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        for path in (self._current_event_path, self._latest_event_path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    def _event_text_fields(
        self, key: str, text: str | None, limit: int = 4000,
    ) -> dict:
        """Store long model outputs in JSONL without letting one event explode."""
        text = text or ""
        truncated = len(text) > limit
        return {
            key: text[:limit],
            f"{key}_truncated": truncated,
            f"{key}_chars": len(text),
        }

    def _entropy_event_fields(
        self, entropies: torch.Tensor, response_mask: torch.Tensor, index: int,
    ) -> dict:
        """Summarize per-token entropy for one generated response."""
        valid = entropies[index][response_mask[index].bool()].detach().float().cpu()
        if valid.numel() == 0:
            return {
                "response_tokens": 0,
                "entropy_mean": 0.0,
                "entropy_min": 0.0,
                "entropy_max": 0.0,
                "entropy_std": 0.0,
            }
        return {
            "response_tokens": int(valid.numel()),
            "entropy_mean": float(valid.mean().item()),
            "entropy_min": float(valid.min().item()),
            "entropy_max": float(valid.max().item()),
            "entropy_std": float(valid.std(unbiased=False).item()),
        }

    def _response_entropy_means(
        self, entropies: torch.Tensor, response_mask: torch.Tensor,
    ) -> list[float]:
        """Mean token entropy for each generated response row."""
        mask_f = response_mask.float()
        h_tensor = (entropies * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
        return [float(x) for x in h_tensor.detach().float().cpu().tolist()]

    def _mean_by_rollout_group(
        self, values: list[float], n_items: int, n_rollouts: int,
    ) -> list[float]:
        """Average flattened per-rollout values back to one value per prompt."""
        grouped: list[float] = []
        for ci in range(n_items):
            start = ci * n_rollouts
            group = values[start:start + n_rollouts]
            grouped.append(float(sum(group) / len(group)) if group else 0.0)
        return grouped

    def _group_entropy_event_fields(
        self,
        entropies: torch.Tensor,
        response_mask: torch.Tensor,
        start_index: int,
        n_rollouts: int,
        per_response_h: list[float],
    ) -> dict:
        """Summarize entropy over all G responses for one prompt."""
        indices = range(start_index, start_index + n_rollouts)
        group_h = per_response_h[start_index:start_index + n_rollouts]
        token_counts: list[int] = []
        valid_chunks: list[torch.Tensor] = []
        for idx in indices:
            valid = entropies[idx][response_mask[idx].bool()].detach().float().cpu()
            token_counts.append(int(valid.numel()))
            if valid.numel() > 0:
                valid_chunks.append(valid)

        if not group_h:
            return {
                "response_tokens": 0,
                "response_tokens_mean": 0.0,
                "entropy_mean": 0.0,
                "entropy_min": 0.0,
                "entropy_max": 0.0,
                "entropy_std": 0.0,
            }

        group_arr = np.asarray(group_h, dtype=float)
        fields = {
            "response_tokens": int(sum(token_counts)),
            "response_tokens_mean": (
                float(sum(token_counts) / len(token_counts)) if token_counts else 0.0
            ),
            "entropy_mean": float(group_arr.mean()),
            "entropy_min": float(group_arr.min()),
            "entropy_max": float(group_arr.max()),
            "entropy_std": float(group_arr.std()),
        }
        if valid_chunks:
            token_values = torch.cat(valid_chunks)
            fields.update({
                "token_entropy_min": float(token_values.min().item()),
                "token_entropy_max": float(token_values.max().item()),
                "token_entropy_std": float(token_values.std(unbiased=False).item()),
            })
        return fields

    def _span_max_entropy_from_response(
        self,
        text: str,
        token_ids,
        entropies: torch.Tensor,
        response_mask: torch.Tensor,
        index: int,
    ) -> float | None:
        """Max mean token entropy over reasoning spans in one response.

        Prefer explicit "Step k:" / "Final:" spans. If the model does not
        format its answer that way, fall back to sentence/newline spans so the
        metric still captures local uncertainty peaks instead of degenerating
        to the whole-response mean.
        """
        valid_mask = response_mask[index].bool().detach().cpu().tolist()
        token_hs_raw = entropies[index].detach().float().cpu().tolist()
        token_hs = [h if m else None for h, m in zip(token_hs_raw, valid_mask)]
        valid_hs = [h for h in token_hs if h is not None]
        if not valid_hs:
            return None

        text = text or ""
        explicit_boundaries = list(re.finditer(r"(?im)^\s*(?:step\s*\d+|final)\s*:", text))

        try:
            ids = token_ids[index].detach().cpu().tolist()
        except Exception:
            try:
                ids = token_ids[index].tolist()
            except Exception:
                return float(sum(valid_hs) / len(valid_hs))

        token_spans = []
        cursor = 0
        for tok_id, h in zip(ids, token_hs):
            if h is None:
                token_spans.append((cursor, cursor))
                continue
            piece = self.tokenizer.decode([tok_id], skip_special_tokens=False)
            if not piece:
                token_spans.append((cursor, cursor))
                continue
            pos = text.find(piece, cursor)
            if pos < 0:
                pos = cursor
            end = pos + len(piece)
            token_spans.append((pos, end))
            cursor = end

        if explicit_boundaries:
            starts = [m.start() for m in explicit_boundaries] + [len(text)]
            spans = list(zip(starts, starts[1:]))
        else:
            # Sentence/newline fallback. This catches common CoT formats that
            # use paragraphs or prose sentences instead of "Step n:" markers.
            spans = []
            start = 0
            for match in re.finditer(r"(?:[.!?]\s+|\n+)", text):
                end = match.end()
                if end > start:
                    spans.append((start, end))
                start = end
            if start < len(text):
                spans.append((start, len(text)))

        span_means = []
        for start, end in spans:
            hs = [
                h for (tok_start, tok_end), h in zip(token_spans, token_hs)
                if h is not None and tok_end > start and tok_start < end
            ]
            if hs:
                span_means.append(sum(hs) / len(hs))
        if not span_means:
            return float(sum(valid_hs) / len(valid_hs))
        return float(max(span_means))

    def _outer_iteration_step(self) -> dict:
        """
        One fixed-budget outer iteration.
        ``inner_iterations`` times, mutate/evaluate one candidate program P'.
        Inner iterations are batched for throughput; there is no early stop.

        순서:
          1. (옵션 A) 기존 champion 재평가 — stale R_Q 갱신
          2. Inner iterations: Mutate → VerifyAndEvaluate → TryInsert
          3. Dataset refresh
        """
        logger.info(
            f"[Evolution] outer iteration at global_step="
            f"{getattr(self, 'global_step', '?')} "
            f"(inner_iterations={self.inner_iterations}, "
            f"inner_iteration_batch_size={self.inner_iteration_batch_size})"
        )
        self._last_event_seq = 0
        self._start_evolution_event_log()

        # ================================================================
        # Phase 0: Champion re-evaluation — Method's self-invalidating archive
        # ================================================================
        # Before mutation, re-score existing champions under the CURRENT Solver
        # so try_insert() compares fresh R_Q against fresh R_Q. Default mode
        # (reeval_per_step=None) re-evaluates every occupied champion.
        n_champions = len(self.map_elites.get_all_champions())
        if self.reeval_per_step is None:
            n_reeval = n_champions
            reeval_mode = "all" if n_champions > 0 else "off"
        elif int(self.reeval_per_step) <= 0:
            n_reeval = 0
            reeval_mode = "off"
        else:
            n_reeval = min(int(self.reeval_per_step), n_champions)
            reeval_mode = "partial" if n_champions > 0 else "off"

        logger.info(
            f"[Reeval] mode={reeval_mode}, targets={n_reeval}/{n_champions} "
            f"occupied champions"
        )
        reeval_metrics = self._reevaluate_champions(n_reeval)
        if reeval_metrics["reevaluated"] > 0 or reeval_metrics["evicted"] > 0:
            logger.info(
                f"[Reeval] updated={reeval_metrics['reevaluated']}, "
                f"evicted={reeval_metrics['evicted']} "
                f"(exec_fail={reeval_metrics.get('exec_fail_evicted', 0)}), "
                f"bin_shifted={reeval_metrics['bin_shifted']}, "
                f"ΔR_Q mean={reeval_metrics['rq_delta_mean']:+.4f}"
            )

        total_attempted = 0
        total_inserted = 0
        inner_iteration_batches = 0

        for inner_start in range(
            1, self.inner_iterations + 1, self.inner_iteration_batch_size
        ):
            inner_iteration_batches += 1
            batch_size = min(
                self.inner_iteration_batch_size,
                self.inner_iterations - inner_start + 1,
            )
            inner_end = inner_start + batch_size - 1
            hard = self.map_elites.count_hard_champions(min_h_bin=2)
            logger.info(
                f"[Evolution] Inner iterations {inner_start}-{inner_end}/"
                f"{self.inner_iterations} (batch_size={batch_size}, H2+={hard})"
            )

            attempted, inserted = self._inner_iteration_batch(
                batch_size=batch_size,
                inner_iteration_start=inner_start,
            )
            total_attempted += attempted
            total_inserted += inserted

        self._refresh_dataset()

        stats = self.map_elites.stats()

        # Champion 분포 통계 (p_hat, H, R_Q)
        champions = self.map_elites.get_all_champions()
        p_hats = [getattr(c, "p_hat", 0.0) for c in champions if c.rq_score]
        h_scores = [getattr(c, "h_score", 0.0) for c in champions if c.rq_score]

        # Frontier-status breakdown (archive material vs. training-eligible).
        # frontier_status is set by _inner_iteration_batch (new candidates) and
        # _reevaluate_champions (updated champions). Champions without the tag
        # (e.g. bootstrap before first reeval) fall through to `frontier`.
        too_hard = too_easy = frontier_cnt = zero_rq = 0
        for c in champions:
            status = (c.metadata or {}).get("frontier_status")
            if status == "too_hard":
                too_hard += 1
            elif status == "too_easy":
                too_easy += 1
            else:
                frontier_cnt += 1
            if not getattr(c, "rq_score", 0.0):
                zero_rq += 1
        refresh_stats = getattr(self, "_last_refresh_stats", {}) or {}

        result = {
            # Inner-iteration stats
            "attempted": total_attempted,
            "inserted": total_inserted,
            "accept_rate": total_inserted / max(total_attempted, 1),
            "inner_iterations": self.inner_iterations,
            "inner_iteration_batches": inner_iteration_batches,
            "inner_iteration_batch_size": self.inner_iteration_batch_size,
            # Legacy key for older dashboards.
            "rounds": inner_iteration_batches,
            # Grid state
            "grid_coverage": stats["coverage"],
            "grid_mean_rq": stats["mean_rq"],
            "grid_max_rq": stats["max_rq"],
            "grid_min_rq": stats["min_rq"],
            "grid_champions": stats["num_champions"],
            "diversity_axis": self.map_elites.diversity_axis,
            "uncertainty_metric": self.uncertainty_metric,
            "reservoir_candidates": stats.get("num_reservoir_candidates", 0),
            "reservoir_cells": stats.get("num_reservoir_cells", 0),
            "reservoir_selections": stats.get("total_reservoir_selections", 0),
            "hard_champions": stats["hard_champions"],
            "total_niches": stats["total_niches"],
            "total_insertions": stats["total_insertions"],
            "total_replacements": stats["total_replacements"],
            "total_reservoir_insertions": stats.get("total_reservoir_insertions", 0),
            "total_duplicate_rejections": stats.get("total_duplicate_rejections", 0),
            # Champion distribution (frontier tracking)
            "champion_p_hat_mean": float(np.mean(p_hats)) if p_hats else 0.0,
            "champion_p_hat_std": float(np.std(p_hats)) if p_hats else 0.0,
            "champion_h_mean": float(np.mean(h_scores)) if h_scores else 0.0,
            "champion_h_std": float(np.std(h_scores)) if h_scores else 0.0,
            # Archive frontier breakdown (p_hat band, for observability)
            "archive_frontier_champions": frontier_cnt,
            "archive_too_hard_champions": too_hard,
            "archive_too_easy_champions": too_easy,
            "archive_zero_rq_champions": zero_rq,
            "dataset_non_frontier_skipped_visits": refresh_stats.get(
                "non_frontier_skipped_visits", 0
            ),
            "dataset_duplicate_signature_skipped_visits": refresh_stats.get(
                "duplicate_signature_skipped_visits", 0
            ),
            # Champion re-evaluation (self-invalidating archive)
            "reeval_count": reeval_metrics["reevaluated"],
            "reeval_evicted": reeval_metrics["evicted"],
            "reeval_exec_fail_evicted": reeval_metrics.get("exec_fail_evicted", 0),
            "reeval_extreme_p_kept": reeval_metrics.get("extreme_p_kept", 0),
            "reeval_bin_shifted": reeval_metrics["bin_shifted"],
            "reeval_rq_delta_mean": reeval_metrics["rq_delta_mean"],
            "reeval_rq_delta_min": reeval_metrics["rq_delta_min"],
            "reeval_rq_delta_max": reeval_metrics["rq_delta_max"],
            "reeval_targets": n_reeval,
            "reeval_n_champions": n_champions,
            # reeval_mode as scalar for wandb compatibility: off=0, partial=1, all=2
            "reeval_mode": {"off": 0, "partial": 1, "all": 2}[reeval_mode],
            # Dataset state
            # True problem count (not __len__, which returns max(n, 1)).
            "dataset_size": len(self.dynamic_dataset.snapshot()),
            "event_log_file": self._current_event_path,
        }

        # Grid 스냅샷 JSON 저장 (시각화는 별도 스크립트로)
        self._save_evolution_snapshot(result)

        return result

    def _evolution_step(self) -> dict:
        """Legacy method name retained for older code paths."""
        return self._outer_iteration_step()

    def _save_evolution_snapshot(self, evo_metrics: dict):
        """매 outer iteration마다 grid 상태 + 챔피언 정보를 JSON으로 저장."""
        import json
        import os
        from datetime import datetime

        save_dir = os.path.join(
            getattr(self.config.trainer, 'default_local_dir', './rq_output'),
            "evolution_logs",
        )
        os.makedirs(save_dir, exist_ok=True)

        step = getattr(self, 'global_step', 0)

        # Grid 스냅샷: 각 셀의 RQ, p_hat, H, 문제 텍스트
        grid_snapshot = []
        for (h, d), niche in self.map_elites.grid.items():
            cell = {"h_bin": h, "div_bin": d, "has_champion": niche.champion is not None}
            if getattr(niche, "candidates", None):
                cell.update({
                    "reservoir_count": len(niche.candidates),
                    "reservoir_best_rq": max(
                        float(getattr(c, "rq_score", 0.0) or 0.0)
                        for c in niche.candidates
                    ),
                    "reservoir_program_ids": [
                        c.program_id for c in niche.candidates[:5]
                    ],
                })
            else:
                cell.update({
                    "reservoir_count": 0,
                    "reservoir_best_rq": 0.0,
                    "reservoir_program_ids": [],
                })
            if niche.champion:
                champ = niche.champion
                inst = champ.execute(seed=0)
                cell.update({
                    "rq_score": champ.rq_score,
                    "p_hat": champ.p_hat,
                    "h_score": champ.h_score,
                    "generation": champ.generation,
                    "program_id": champ.program_id,
                    "root_seed_id": champ.root_seed_id,
                    "concept_type": champ.get_concept_type(),
                    "concept_group": champ.get_concept_group(),
                    "uncertainty_metric": (champ.metadata or {}).get(
                        "uncertainty_metric", self.uncertainty_metric
                    ),
                    "uncertainty_scores": (champ.metadata or {}).get(
                        "uncertainty_scores", {}
                    ),
                    "token_entropy": (champ.metadata or {}).get("token_entropy"),
                    "problem": inst.problem if inst else "",
                    "answer": inst.answer if inst else "",
                    "source_code": champ.source_code,
                })
            grid_snapshot.append(cell)

        snapshot = {
            "global_step": step,
            "outer_iteration_index": self._evolution_event_seq,
            # Legacy key for existing analysis scripts.
            "evolution_index": self._evolution_event_seq,
            "timestamp": datetime.now().isoformat(),
            "metrics": evo_metrics,
            "grid": grid_snapshot,
            "diversity_axis": self.map_elites.diversity_axis,
            "uncertainty_metric": self.uncertainty_metric,
            "seed_labels": {
                str(d): label
                for d, label in enumerate(self.map_elites.diversity_labels())
            },
        }

        # step별 파일 저장
        path = os.path.join(save_dir, f"evo_step_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

        # 최신 스냅샷 덮어쓰기 (빠른 접근용)
        latest_path = os.path.join(save_dir, "latest.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"[Evolution] Snapshot saved: {path}")

    def _save_dataset_snapshot(
        self, problems: list[dict], frontier_champs: int, archive_champs: int,
    ) -> None:
        """Persist the actual training problem list produced by a refresh.

        Written under {default_local_dir}/evolution_logs/datasets/ :
          - dataset_step_{global_step}.json — step-indexed
          - latest_dataset.json            — always the most recent

        The problem dicts already include problem / answer / program_id /
        rq_score / h_bin / d_bin / seed, so a downstream consumer can
        reconstruct which archive cell each training example came from.
        """
        import json
        import os
        from datetime import datetime

        save_dir = os.path.join(
            getattr(self.config.trainer, 'default_local_dir', './rq_output'),
            "evolution_logs",
            "datasets",
        )
        os.makedirs(save_dir, exist_ok=True)

        step = getattr(self, 'global_step', 0)
        refresh_stats = getattr(self, "_last_refresh_stats", {}) or {}
        snapshot = {
            "global_step": step,
            "outer_iteration_index": self._evolution_event_seq,
            # Legacy key for existing analysis scripts.
            "evolution_index": self._evolution_event_seq,
            "timestamp": datetime.now().isoformat(),
            "training_selection_mode": self.training_selection_mode,
            "training_budget": self.training_budget,
            "instances_per_program": self.instances_per_program,
            "strict_anti_reuse": self.strict_anti_reuse,
            "frontier_p_hat_range": list(self.frontier_p_hat_range),
            "dataset_size": len(problems),
            "archive_champions": archive_champs,
            "frontier_champions": frontier_champs,
            "duplicate_signature_skipped_visits": refresh_stats.get(
                "duplicate_signature_skipped_visits", 0
            ),
            "problems": problems,
        }

        path = os.path.join(save_dir, f"dataset_step_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

        latest_path = os.path.join(save_dir, "latest_dataset.json")
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

        logger.info(
            f"[Evolution] Dataset snapshot saved: {path} "
            f"({len(problems)} problems, "
            f"frontier={frontier_champs}/{archive_champs} champs)"
        )

    # ------------------------------------------------------------------
    # Checkpoint override — persist used_seeds across restart
    # ------------------------------------------------------------------

    _USED_SEEDS_FILE = "rq_used_seeds.json"

    def _save_checkpoint(self) -> None:
        """Persist used_seeds alongside actor/critic/dataloader state.

        strict_anti_reuse=True guarantees "(program_id, seed) used at most once
        across the entire run" — but only if used_seeds survives restart. This
        hook writes a JSON sidecar to the checkpoint folder; the base class
        saves actor/critic/dataloader as usual.
        """
        import json
        import os

        super()._save_checkpoint()

        folder_path = os.path.join(
            self.config.trainer.save_checkpoint_path,
            f"global_step_{self.global_step}",
        )
        used_seeds_path = os.path.join(folder_path, self._USED_SEEDS_FILE)
        # Sets are not JSON-serialisable — dump as sorted lists so diffs stay
        # readable and round-trip is stable.
        payload = {
            "strict_anti_reuse": self.strict_anti_reuse,
            "instances_per_program": self.instances_per_program,
            "used_seeds": {
                pid: sorted(seeds) for pid, seeds in self.used_seeds.items()
            },
        }
        try:
            with open(used_seeds_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(
                f"[Checkpoint] Saved used_seeds ({len(self.used_seeds)} programs, "
                f"{sum(len(v) for v in self.used_seeds.values())} pairs) → "
                f"{used_seeds_path}"
            )
        except Exception as exc:
            logger.warning(f"[Checkpoint] Failed to save used_seeds: {exc}")

    def _load_checkpoint(self) -> None:
        """Restore used_seeds after the base-class checkpoint load."""
        import json
        import os

        super()._load_checkpoint()

        if self.config.trainer.load_checkpoint_path is None:
            return

        used_seeds_path = os.path.join(
            self.config.trainer.load_checkpoint_path, self._USED_SEEDS_FILE,
        )
        if not os.path.exists(used_seeds_path):
            logger.info(
                f"[Checkpoint] No used_seeds sidecar at {used_seeds_path}; "
                f"starting with empty used_seeds. strict_anti_reuse cannot "
                f"protect against pre-checkpoint collisions."
            )
            return

        try:
            with open(used_seeds_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            raw = payload.get("used_seeds", {}) or {}
            self.used_seeds = {pid: set(seeds) for pid, seeds in raw.items()}
            logger.info(
                f"[Checkpoint] Restored used_seeds: {len(self.used_seeds)} "
                f"programs, {sum(len(v) for v in self.used_seeds.values())} "
                f"pairs from {used_seeds_path}"
            )
        except Exception as exc:
            logger.warning(
                f"[Checkpoint] Failed to load used_seeds from "
                f"{used_seeds_path}: {exc}"
            )

    # ------------------------------------------------------------------
    # Option A — Champion re-evaluation
    # ------------------------------------------------------------------

    def _reevaluate_champions(self, n_reeval: int) -> dict:
        """
        기존 champion 중 n_reeval 개를 현재 solver 로 다시 점수 매김.

        동기:
          try_insert() 는 저장된 R_Q 와 새 후보 R_Q 를 단순 비교하므로,
          champion 의 R_Q 가 진화 당시 solver 기준 (stale) 이면 공정한 비교가
          어렵다. Solver 가 학습되면서 이미 마스터한 champion 이 계속
          dataset 에 재주입되어 learnability plateau 를 유발한다.

          이 메서드는 매 outer iteration 앞에서:
            - age-weighted 로 champion 샘플링 (오래 재평가 안 된 것 우선)
            - fresh seed 로 instance 재생성
            - 현재 solver 로 p_hat / H 측정 → R_Q 갱신
            - p_hat 극단인 champion 은 archive material 로 유지하되
              Solver training data 에서는 제외
            - H 재측정으로 bin 경계를 넘으면 rebin_champion() 으로 이동

        Returns:
            metrics dict — reevaluated / evicted / bin_shifted /
                           rq_delta_{mean,min,max}
        """
        # All counters initialized at function entry so early-return paths
        # (no champions, all execute-failing) carry the evictions they did.
        evicted = 0
        exec_fail_evicted = 0
        extreme_p_kept = 0
        bin_shifted = 0
        reevaluated = 0
        rq_deltas: list[float] = []

        def _metrics() -> dict:
            return {
                "reevaluated": reevaluated,
                "evicted": evicted,
                "exec_fail_evicted": exec_fail_evicted,
                "extreme_p_kept": extreme_p_kept,
                "bin_shifted": bin_shifted,
                "rq_delta_mean": float(np.mean(rq_deltas)) if rq_deltas else 0.0,
                "rq_delta_min": float(np.min(rq_deltas)) if rq_deltas else 0.0,
                "rq_delta_max": float(np.max(rq_deltas)) if rq_deltas else 0.0,
            }

        if n_reeval <= 0:
            return _metrics()

        champions = self.map_elites.get_all_champions()
        if not champions:
            return _metrics()

        current_step = getattr(self, "global_step", 0)

        # ---- 1. Target selection ----
        # n_reeval >= len(champions) is the Method-aligned "all" mode — bypass
        # age-weighted sampling so no champion is silently skipped.
        if n_reeval >= len(champions):
            targets = list(champions)
        else:
            n = n_reeval
            champions_by_age = sorted(
                champions,
                key=lambda c: getattr(c, "last_reeval_step", -1),
            )
            n_old = int(n * self.reeval_age_ratio)
            n_rand = n - n_old

            targets = list(champions_by_age[:n_old])
            remaining = [c for c in champions if c not in targets]
            if remaining and n_rand > 0:
                targets += random.sample(remaining, min(n_rand, len(remaining)))
            targets = targets[:n]

        # ---- 2. Fresh seed 로 instance 재생성 ----
        # Retry up to MAX_EXEC_RETRIES seeds per champion; evict if all fail.
        # Without this, an exec-failing champion would silently keep its stale
        # R_Q in the archive — contradicting the self-invalidating claim.
        MAX_EXEC_RETRIES = 5
        pairs: list[tuple[ProblemProgram, ProblemInstance, float]] = []
        for champ in targets:
            inst = None
            for _ in range(MAX_EXEC_RETRIES):
                seed = random.randint(0, 10_000)
                inst = champ.execute(seed=seed)
                if inst is not None:
                    break
            if inst is not None:
                pairs.append((champ, inst, float(champ.rq_score or 0.0)))
            else:
                if self.map_elites.evict_champion(champ):
                    evicted += 1
                    exec_fail_evicted += 1
                    logger.info(
                        f"[Reeval] EVICT exec-failing ({champ.niche_h},"
                        f"{champ.niche_div}) id={champ.program_id} "
                        f"(execute returned None for {MAX_EXEC_RETRIES} seeds)"
                    )

        if not pairs:
            return _metrics()

        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        temperature = self.config.worker.rollout.temperature

        # Solver prompt 구성 (기존 inner-iteration scoring 과 동일)
        solver_texts = []
        for _, inst, _ in pairs:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": inst.problem},
            ]
            if self.tokenizer.chat_template:
                solver_texts.append(self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                ))
            else:
                solver_texts.append(
                    f"system: {SYSTEM_PROMPT}\nuser: {inst.problem}"
                )

        # ---- 3. Full-G rollout + full-vocab entropy -------------------
        # Every generated response contributes both a correctness sample and
        # a token-entropy sample. This keeps U aligned with the same G
        # trajectories used for p_hat, instead of special-casing rollout 0.
        n_pairs = len(pairs)
        pair_answers = [inst.answer for _, inst, _ in pairs]
        n_eval_rollouts = max(1, int(self.num_rollouts))

        rollout_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=solver_texts,
            answers=pair_answers,
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
            n_repeat=n_eval_rollouts,
        )
        try:
            rollout_output = self._generate_sequences_dp_safe(rollout_batch)
        except Exception as e:
            logger.warning(f"[Reeval] rollout failed: {e}")
            return _metrics()

        rollout_resp_ids = rollout_output.batch.get("responses")
        response_mask = rollout_output.batch.get("response_mask")
        if rollout_resp_ids is None or response_mask is None:
            return _metrics()

        expected_rows = n_pairs * n_eval_rollouts
        if rollout_resp_ids.shape[0] != expected_rows:
            logger.warning(
                f"[Reeval] rollout rows {rollout_resp_ids.shape[0]} != "
                f"expected {expected_rows}"
            )
            return _metrics()

        flags_per_pair: list[list[bool]] = [[] for _ in range(n_pairs)]
        predictions_per_pair: list[list[str | None]] = [[] for _ in range(n_pairs)]
        decoded_per_pair: list[list[str]] = [[] for _ in range(n_pairs)]
        for ci in range(n_pairs):
            ans = pair_answers[ci]
            for ri in range(n_eval_rollouts):
                idx = ci * n_eval_rollouts + ri
                decoded = self.tokenizer.decode(
                    rollout_resp_ids[idx].tolist(), skip_special_tokens=True,
                )
                pred = _extract_boxed(decoded)
                decoded_per_pair[ci].append(decoded)
                predictions_per_pair[ci].append(pred)
                flags_per_pair[ci].append(_answers_match(pred, ans) if pred else False)

        try:
            entropy_output = self._compute_log_probs_dp_safe(
                rollout_output, calculate_entropy=True, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[Reeval] entropy forward failed: {e}")
            return _metrics()

        entropies = entropy_output.batch.get("entropies")
        if entropies is None:
            logger.warning("[Reeval] actor did not return entropies")
            return _metrics()
        assert entropies.shape == response_mask.shape, (
            f"[Reeval] entropies {tuple(entropies.shape)} != response_mask "
            f"{tuple(response_mask.shape)}"
        )

        h_per_response = self._response_entropy_means(entropies, response_mask)
        h_per_pair = self._mean_by_rollout_group(
            h_per_response, n_pairs, n_eval_rollouts,
        )
        span_max_per_pair: list[float | None] = []
        for ci in range(n_pairs):
            span_values = []
            for ri in range(n_eval_rollouts):
                idx = ci * n_eval_rollouts + ri
                span_i = self._span_max_entropy_from_response(
                    decoded_per_pair[ci][ri],
                    rollout_resp_ids,
                    entropies,
                    response_mask,
                    idx,
                )
                if span_i is not None:
                    span_values.append(span_i)
            span_max_per_pair.append(
                float(sum(span_values) / len(span_values)) if span_values else None
            )

        # ---- 4. Counters (evicted / exec_fail_evicted already initialized) --
        # frontier_p_hat_range defines the training-data frontier band; it is
        # NOT a reeval eviction gate anymore. Extreme-p champions stay in the
        # archive as mutation material and are excluded from training via
        # _refresh_dataset(); their R_Q naturally collapses to 0.
        low, high = self.frontier_p_hat_range
        extreme_p_kept = 0

        # ---- 5. Update / rebin — NEVER evict on extreme-p anymore ------
        # p=0 / p=1 champions are kept as archive material and will be
        # filtered out of Solver training data by _refresh_dataset().
        # Their R_Q naturally drops to 0 (since p(1-p)=0), so they cannot
        # displace positive-R_Q candidates at try_insert time.
        for ci in range(n_pairs):
            champ, inst, rq_before = pairs[ci]
            h_bar = h_per_pair[ci]
            flags = flags_per_pair[ci]
            pred_answers = predictions_per_pair[ci]
            new_p_hat = sum(flags) / len(flags) if flags else 0.0
            uncertainty_score, uncertainty_scores = _select_uncertainty_score(
                self.uncertainty_metric,
                h_bar,
                flags,
                pred_answers,
                span_max_per_pair[ci],
            )
            new_rq = new_p_hat * (1.0 - new_p_hat) * uncertainty_score

            champ.p_hat = new_p_hat
            champ.h_score = uncertainty_score
            champ.rq_score = new_rq
            champ.fitness = new_rq
            champ.last_reeval_step = current_step
            champ.metadata["uncertainty_metric"] = self.uncertainty_metric
            champ.metadata["uncertainty_scores"] = uncertainty_scores
            champ.metadata["token_entropy"] = h_bar
            champ.metadata["frontier_status"] = (
                "too_hard" if new_p_hat <= low
                else "too_easy" if new_p_hat >= high
                else "frontier"
            )
            if new_p_hat <= low or new_p_hat >= high:
                extreme_p_kept += 1

            moved = self.map_elites.rebin_champion(
                program=champ, new_h_value=uncertainty_score, problem_text=inst.problem,
            )
            if moved:
                bin_shifted += 1

            rq_deltas.append(float(champ.rq_score) - rq_before)
            reevaluated += 1

        if extreme_p_kept:
            logger.info(
                f"[Reeval] kept {extreme_p_kept} extreme-p champions in archive "
                f"(mutation material; excluded from training dataset)"
            )
        return _metrics()

    def _inner_iteration_batch(
        self, batch_size: int, inner_iteration_start: int,
    ) -> tuple[int, int]:
        """
        Batched inner iterations.

        논문 표기상 inner iteration 하나는 candidate program P' 하나의
        mutation → rollout → entropy → grid insert 이다. 구현에서는
        throughput을 위해 여러 inner iteration을 한 번에 batch 처리한다.
        _refresh_dataset()는 호출하지 않음 (caller가 최종 1회 호출).

        Returns: (attempted, inserted)
        """
        inner_iteration_end = inner_iteration_start + batch_size - 1
        legacy_batch_index = (
            (inner_iteration_start - 1) // self.inner_iteration_batch_size
        ) + 1
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        temperature = self.config.worker.rollout.temperature

        # ================================================================
        # Phase 1: Mutation — 연산자 선택 + batch generate
        # ================================================================

        few_shot = build_few_shot_examples(self.map_elites, top_k=3)

        mutation_prompts = []  # (prompt_text, op, parent, parent_b, recovery_prefix)
        for _ in range(batch_size):
            roll = random.random()
            if roll < self.crossover_ratio:
                op = "crossover"
            elif roll < self.crossover_ratio + self.in_depth_ratio:
                op = "in_depth"
            else:
                op = "in_breadth"

            if op == "crossover":
                pa, pb = self.map_elites.sample_two_parents()
                if pa is None or pb is None:
                    pa = self.map_elites.sample_parent()
                    if pa is None:
                        continue
                    op = "in_depth"
                else:
                    prompt_text = MUTATE_CROSSOVER.format(
                        code_a=pa.source_code, code_b=pb.source_code,
                        p_hat_a=getattr(pa, "p_hat", 0.5),
                        h_a=getattr(pa, "h_score", 1.0),
                        p_hat_b=getattr(pb, "p_hat", 0.5),
                        h_b=getattr(pb, "h_score", 1.0),
                        few_shot=few_shot,
                    )
                    chosen_t, chosen_g = choose_prefill_concept(op, pa, pb)
                    suffix, recovery = build_mutation_prefill(
                        chosen_t, chosen_g, op=op
                    )
                    full_prompt = _format_mutation_prompt(
                        self.tokenizer, prompt_text, suffix
                    )
                    mutation_prompts.append(
                        (full_prompt, op, pa, pb, recovery)
                    )
                    continue

            parent = self.map_elites.sample_parent() if op != "crossover" else pa
            if parent is None:
                continue

            score_fb = build_score_feedback(parent)
            exec_fb = build_execution_feedback(parent)
            ctype, cgroup, suggested = parent_concept_fields(parent)
            tmpl = MUTATE_DEPTH if op == "in_depth" else MUTATE_BREADTH
            prompt_text = tmpl.format(
                code=parent.source_code,
                score_feedback=score_fb,
                exec_feedback=exec_fb,
                few_shot=few_shot,
                parent_concept_type=ctype,
                parent_concept_group=cgroup,
                suggested_groups=suggested,
            )
            chosen_t, chosen_g = choose_prefill_concept(op, parent)
            suffix, recovery = build_mutation_prefill(
                chosen_t, chosen_g, op=op
            )
            full_prompt = _format_mutation_prompt(
                self.tokenizer, prompt_text, suffix
            )
            mutation_prompts.append(
                (full_prompt, op, parent, None, recovery)
            )

        if not mutation_prompts:
            self._record_evolution_event({
                "event": "inner_iteration_batch_empty",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "reason": "no_parent",
            })
            return 0, 0

        mut_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=[pt for pt, _, _, _, _ in mutation_prompts],
            answers=[""] * len(mutation_prompts),
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
        )

        try:
            mut_output = self._generate_sequences_dp_safe(mut_batch)
        except Exception as e:
            logger.warning(f"[Evolution] batch mutation failed: {e}")
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "mutation_generate",
                "error": str(e),
            })
            return 0, 0

        resp_ids = mut_output.batch.get("responses")
        if resp_ids is None:
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "mutation_generate",
                "error": "responses missing",
            })
            return 0, 0

        # Decode + execute (CPU)
        children = []  # (child, inst, op, mutation_output, source_code, inner_iteration)
        for i, (_, op, parent, parent_b, recovery) in enumerate(mutation_prompts):
            inner_iteration = inner_iteration_start + i
            code_text = self.tokenizer.decode(
                resp_ids[i].tolist(), skip_special_tokens=True
            )
            source_code = _extract_code(recovery + code_text)
            if not source_code:
                self._record_evolution_event({
                    "event": "candidate_failed",
                    "round": legacy_batch_index,
                    "inner_iteration": inner_iteration,
                    "candidate_index": i,
                    "op": op,
                    "parent_id": parent.program_id,
                    "parent_b_id": parent_b.program_id if parent_b is not None else None,
                    "status": "no_code",
                    **self._event_text_fields("mutation_output", code_text, limit=8000),
                })
                continue

            if op == "crossover" and parent_b is not None:
                child = ProblemProgram(
                    source_code=source_code,
                    parent_id=f"{parent.program_id}×{parent_b.program_id}",
                    generation=max(parent.generation, parent_b.generation) + 1,
                    metadata={"op": "crossover"},
                )
            else:
                child = ProblemProgram(
                    source_code=source_code,
                    parent_id=parent.program_id,
                    generation=parent.generation + 1,
                    metadata={"op": op},
                )

            # Multi-seed + SymPy 자가 검증
            inst = _verify_program(child, n_seeds=5)
            if inst is None:
                self._record_evolution_event({
                    "event": "candidate_failed",
                    "round": legacy_batch_index,
                    "inner_iteration": inner_iteration,
                    "candidate_index": i,
                    "op": op,
                    "parent_id": parent.program_id,
                    "parent_b_id": parent_b.program_id if parent_b is not None else None,
                    "child_id": child.program_id,
                    "status": "verify_failed",
                    **self._event_text_fields("mutation_output", code_text, limit=8000),
                    **self._event_text_fields("source_code", source_code, limit=12000),
                })
                continue
            parent_type = parent.get_concept_type() if parent is not None else None
            parent_group = parent.get_concept_group() if parent is not None else None
            child_type = child.get_concept_type()
            child_group = child.get_concept_group()
            concept_op_reason = None
            if op == "in_depth" and parent_type and child_type != parent_type:
                concept_op_reason = (
                    f"in_depth must preserve CONCEPT_TYPE "
                    f"({parent_type} -> {child_type})"
                )
            elif op == "in_breadth" and parent_group and child_group == parent_group:
                concept_op_reason = (
                    f"in_breadth must change CONCEPT_GROUP "
                    f"(stayed {child_group})"
                )
            if concept_op_reason:
                self._record_evolution_event({
                    "event": "candidate_failed",
                    "round": legacy_batch_index,
                    "inner_iteration": inner_iteration,
                    "candidate_index": i,
                    "op": op,
                    "parent_id": parent.program_id,
                    "parent_b_id": parent_b.program_id if parent_b is not None else None,
                    "child_id": child.program_id,
                    "status": "concept_operator_reject",
                    "failure_reason": concept_op_reason,
                    "concept_type": child_type,
                    "concept_group": child_group,
                    "problem": inst.problem,
                    "answer": inst.answer,
                    **self._event_text_fields("mutation_output", code_text, limit=8000),
                    **self._event_text_fields("source_code", source_code, limit=12000),
                })
                continue
            self._record_evolution_event({
                "event": "candidate_verified",
                "round": legacy_batch_index,
                "inner_iteration": inner_iteration,
                "candidate_index": i,
                "op": op,
                "parent_id": parent.program_id,
                "parent_b_id": parent_b.program_id if parent_b is not None else None,
                "child_id": child.program_id,
                "generation": child.generation,
                "concept_type": child.get_concept_type(),
                "concept_group": child.get_concept_group(),
                "problem": inst.problem,
                "answer": inst.answer,
                **self._event_text_fields("mutation_output", code_text, limit=8000),
                **self._event_text_fields("source_code", source_code, limit=12000),
            })
            children.append((child, inst, op, code_text, source_code, inner_iteration))

        if not children:
            self._record_evolution_event({
                "event": "inner_iteration_batch_empty",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "reason": "no_verified_children",
                "mutation_prompts": len(mutation_prompts),
            })
            return 0, 0

        # ================================================================
        # Phase 2-4: full-G rollout entropy → R_Q
        # ================================================================
        # 각 검증 통과 후보마다 G개 response를 생성하고, 그 G개 전부에 대해
        # actor forward로 full-vocab entropy를 계산한다. p_hat과 U가 같은
        # rollout set에서 추정되므로 probe 특례가 사라진다.
        solver_texts = []
        for _, inst, _, _, _, _ in children:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": inst.problem}]
            if self.tokenizer.chat_template:
                solver_texts.append(self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
            else:
                solver_texts.append(f"system: {SYSTEM_PROMPT}\nuser: {inst.problem}")

        n_children = len(children)
        inst_answers = [inst.answer for _, inst, _, _, _, _ in children]
        n_eval_rollouts = max(1, int(self.num_rollouts))

        # ---- (a) Generate G rollouts at rollout temperature ------------
        rollout_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=solver_texts,
            answers=inst_answers,
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
            n_repeat=n_eval_rollouts,
        )
        try:
            rollout_output = self._generate_sequences_dp_safe(rollout_batch)
        except Exception as e:
            logger.warning(f"[Evolution] rollout failed: {e}")
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "rollout",
                "error": str(e),
            })
            return 0, 0

        rollout_resp_ids = rollout_output.batch.get("responses")
        response_mask = rollout_output.batch.get("response_mask")
        if rollout_resp_ids is None or response_mask is None:
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "rollout",
                "error": "responses or response_mask missing",
            })
            return 0, 0

        expected_rows = n_children * n_eval_rollouts
        if rollout_resp_ids.shape[0] != expected_rows:
            err = (
                f"rollout rows {rollout_resp_ids.shape[0]} != "
                f"expected {expected_rows}"
            )
            logger.warning(f"[Evolution] {err}")
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "rollout",
                "error": err,
            })
            return 0, 0

        flags_per_child: list[list[bool]] = [[] for _ in range(n_children)]
        predictions_per_child: list[list[str | None]] = [[] for _ in range(n_children)]
        decoded_per_child: list[list[str]] = [[] for _ in range(n_children)]
        for ci in range(n_children):
            ans = inst_answers[ci]
            for ri in range(n_eval_rollouts):
                idx = ci * n_eval_rollouts + ri
                decoded = self.tokenizer.decode(
                    rollout_resp_ids[idx].tolist(), skip_special_tokens=True
                )
                pred = _extract_boxed(decoded)
                decoded_per_child[ci].append(decoded)
                predictions_per_child[ci].append(pred)
                flags_per_child[ci].append(_answers_match(pred, ans) if pred else False)

        # ---- (b) Full-vocab entropy via actor forward pass -------------
        try:
            entropy_output = self._compute_log_probs_dp_safe(
                rollout_output, calculate_entropy=True, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[Evolution] entropy forward failed: {e}")
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "entropy_forward",
                "error": str(e),
            })
            return 0, 0

        entropies = entropy_output.batch.get("entropies")
        if entropies is None:
            logger.warning("[Evolution] actor did not return entropies")
            self._record_evolution_event({
                "event": "inner_iteration_batch_failed",
                "round": legacy_batch_index,
                "inner_iteration_start": inner_iteration_start,
                "inner_iteration_end": inner_iteration_end,
                "phase": "entropy_forward",
                "error": "entropies missing",
            })
            return 0, 0
        assert entropies.shape == response_mask.shape, (
            f"[Evolution] entropies {tuple(entropies.shape)} != response_mask "
            f"{tuple(response_mask.shape)}"
        )

        h_per_response = self._response_entropy_means(entropies, response_mask)
        h_per_child = self._mean_by_rollout_group(
            h_per_response, n_children, n_eval_rollouts,
        )
        entropy_event_fields = [
            self._group_entropy_event_fields(
                entropies,
                response_mask,
                ci * n_eval_rollouts,
                n_eval_rollouts,
                h_per_response,
            )
            for ci in range(n_children)
        ]
        span_max_per_child: list[float | None] = []
        for ci in range(n_children):
            span_values = []
            for ri in range(n_eval_rollouts):
                idx = ci * n_eval_rollouts + ri
                span_i = self._span_max_entropy_from_response(
                    decoded_per_child[ci][ri],
                    rollout_resp_ids,
                    entropies,
                    response_mask,
                    idx,
                )
                if span_i is not None:
                    span_values.append(span_i)
            span_max_per_child.append(
                float(sum(span_values) / len(span_values)) if span_values else None
            )

        # ---- (d) Score + grid insertion ------------------------------
        # p=0 / p=1 candidates are KEPT as archive material (evolutionary
        # parents, empty-niche fillers). They produce R_Q=0 so they cannot
        # displace a positive-R_Q champion, and _refresh_dataset() skips
        # them when assembling Solver training data. This follows Method's
        # pseudocode (TryInsert has no p_hat gate) while keeping Solver
        # training focused on the learnability frontier.
        f_low, f_high = self.frontier_p_hat_range
        attempted = n_children
        inserted = 0
        for ci in range(n_children):
            child, inst, op, code_text, source_code, inner_iteration = children[ci]
            h_bar = h_per_child[ci]
            flags = flags_per_child[ci]
            pred_answers = predictions_per_child[ci]
            p_hat = sum(flags) / len(flags) if flags else 0.0
            uncertainty_score, uncertainty_scores = _select_uncertainty_score(
                self.uncertainty_metric,
                h_bar,
                flags,
                pred_answers,
                span_max_per_child[ci],
            )

            rq_result = compute_rq_full(flags, uncertainty_score)
            target_h = self.map_elites.h_to_bin(uncertainty_score)
            target_d = self.map_elites.program_to_div_bin(child, inst.problem)
            target_niche = self.map_elites.grid.get((target_h, target_d))
            previous_program_id = (
                target_niche.champion.program_id
                if target_niche and target_niche.champion is not None
                else None
            )
            previous_rq = (
                target_niche.champion_rq
                if target_niche and target_niche.champion is not None
                else None
            )
            child.p_hat = p_hat
            child.h_score = uncertainty_score
            child.rq_score = rq_result.rq_score
            child.fitness = rq_result.rq_score
            child.metadata["uncertainty_metric"] = self.uncertainty_metric
            child.metadata["uncertainty_scores"] = uncertainty_scores
            child.metadata["token_entropy"] = h_bar
            child.metadata["frontier_status"] = (
                "too_hard" if p_hat <= f_low
                else "too_easy" if p_hat >= f_high
                else "frontier"
            )

            was_inserted = self.map_elites.try_insert(
                program=child, h_value=uncertainty_score,
                problem_text=inst.problem, rq_score=rq_result.rq_score,
            )
            first_prediction = pred_answers[0] if pred_answers else None
            first_correct = flags[0] if flags else False
            first_response = decoded_per_child[ci][0] if decoded_per_child[ci] else ""
            self._record_evolution_event({
                "event": "candidate_scored",
                "round": legacy_batch_index,
                "inner_iteration": inner_iteration,
                "candidate_index": ci,
                "op": op,
                "child_id": child.program_id,
                "parent_id": child.parent_id,
                "status": "inserted" if was_inserted else "rejected",
                "reservoir_saved": (
                    (not was_inserted)
                    and child.metadata.get("archive_status") == "reservoir"
                ),
                "reservoir_reason": child.metadata.get("reservoir_reason"),
                "duplicate_of": child.metadata.get("duplicate_of"),
                "archive_status": child.metadata.get("archive_status"),
                "frontier_status": child.metadata["frontier_status"],
                "h_bar": uncertainty_score,
                "token_entropy": h_bar,
                "uncertainty_metric": self.uncertainty_metric,
                "uncertainty_scores": uncertainty_scores,
                "p_hat": p_hat,
                "rq_score": rq_result.rq_score,
                "h_bin": target_h,
                "d_bin": target_d,
                "concept_type": child.get_concept_type(),
                "concept_group": child.get_concept_group(),
                "probe_prediction": first_prediction,
                "probe_correct": first_correct,
                **self._event_text_fields("probe_response", first_response),
                **self._event_text_fields("mutation_output", code_text, limit=8000),
                **self._event_text_fields("source_code", source_code, limit=12000),
                **entropy_event_fields[ci],
                "num_rollouts": len(flags),
                "num_correct": sum(flags),
                "previous_program_id": previous_program_id,
                "previous_rq": previous_rq,
                "problem": inst.problem,
                "answer": inst.answer,
            })
            if was_inserted:
                inserted += 1
                logger.info(
                    f"[Evolution] Inserted ({op}, "
                    f"{child.metadata['frontier_status']}): p={p_hat:.3f} "
                    f"U[{self.uncertainty_metric}]={uncertainty_score:.3f} "
                    f"R_Q={rq_result.rq_score:.4f}"
                )

        return attempted, inserted

    def _evolution_round(self, batch_size: int, round_num: int) -> tuple[int, int]:
        """Legacy method name retained for older tests/scripts."""
        inner_iteration_start = ((round_num - 1) * batch_size) + 1
        return self._inner_iteration_batch(batch_size, inner_iteration_start)

    # ------------------------------------------------------------------
    # Dataset refresh from MAP-Elites champions
    # ------------------------------------------------------------------

    def _next_unused_seed(self, program_id: str) -> int:
        """Smallest non-negative integer not in ``used_seeds[program_id]``.

        No cap. ``instances_per_program`` is a per-refresh emission limit
        enforced in :meth:`_refresh_dataset` via a local counter — it is NOT
        a lifetime seed-space bound for a champion. Without this separation,
        a long-lived strict-mode champion would run out of seeds after
        ``instances_per_program`` refreshes even though the program generator
        accepts arbitrary integer seeds.

        ``strict_anti_reuse`` semantics:
          True  — used set persists across refreshes (and across runs via
                  the checkpoint sidecar), so seeds monotonically increase
                  per program: refresh 0 → 0..15, refresh 1 → 16..31, …
          False — :meth:`_refresh_dataset` clears ``self.used_seeds`` at
                  refresh start, so seeds restart at 0 each refresh.

        Always returns a valid seed (integer space is unbounded).
        """
        used = self.used_seeds.setdefault(program_id, set())
        s = 0
        while s in used:
            s += 1
        return s

    def _is_frontier_champion(self, champ) -> bool:
        """Whether this champion is in the learnability frontier for training.

        p=0 / p=1 champions are kept in the archive as evolutionary material
        (parents, empty-niche fillers) but excluded here from Solver training
        data. This follows MAP-Elites exploration semantics: empty niches may
        store zero-R_Q programs, while training stays on the frontier.

        p_hat missing (e.g. seed bootstrap before any reeval) → treated as
        frontier so the very first outer iteration's dataset isn't silently emptied.
        """
        p = getattr(champ, "p_hat", None)
        if p is None:
            return True
        low, high = self.frontier_p_hat_range
        return float(low) < float(p) < float(high)

    def _refresh_dataset(self):
        """MAP-Elites 챔피언 → dynamic_dataset 교체 → dataloader 재구성.

        ``instances_per_program`` is a **per-refresh** emission cap, NOT a
        lifetime seed-space bound. Seeds themselves are unbounded — a strict
        champion surviving K refreshes contributes seeds
        ``0..K·instances_per_program − 1`` in monotonic order across the run.

        Selection rule (``h_priority_d_uniform``):
          Repeated H×D sweeps emit ≤1 new problem per filled niche per sweep.
          Across sweeps of the same refresh, the same champion may be
          revisited with its next unused seed until it has emitted
          ``instances_per_program`` instances this refresh.
          Caps therefore stack as:

              dataset_size ≤ min(
                  training_budget,
                  frontier_champions × instances_per_program,
              )

          Examples (instances_per_program=16, budget=960):
              frontier=1  → dataset ≤ 16
              frontier=10 → dataset ≤ 160
              frontier=60 → dataset ≤ 960 (budget saturates)

          Seeds across refreshes, strict_anti_reuse=True:
              refresh 0 : prog A uses seeds 0..15
              refresh 1 : prog A uses seeds 16..31
              refresh 2 : prog A uses seeds 32..47
              …
          Seeds across refreshes, strict_anti_reuse=False:
              every refresh restarts from seed 0.

          Only frontier champions (f_low < p_hat < f_high) contribute to
          training data; extreme-p champions stay in the archive as mutation
          material.

          ``training_budget`` under-fill is NORMAL for small archives and is
          emitted as an info-level log rather than an error.

        Legacy ``uniform`` mode:
          Iterates every frontier champion up to ``instances_per_program``
          emissions each refresh, then clamps to ``training_budget``.
        """
        old_size = len(self.dynamic_dataset.snapshot())
        new_problems: list[dict] = []
        sweep = 0
        non_frontier_skipped = 0  # reporting-only counter

        # Non-strict anti-reuse: clear the used-seed registry so the same
        # (program_id, seed) pair may recur across outer iterations.
        # Strict mode keeps used_seeds so seeds monotonically advance.
        if not self.strict_anti_reuse:
            self.used_seeds.clear()

        # Per-refresh emission counter: caps how many instances a single
        # champion contributes to *this* refresh's dataset. Reset each call.
        # This is the ONLY place instances_per_program acts as a cap; seed
        # space itself is unbounded (see _next_unused_seed docstring).
        emitted_per_program: dict[str, int] = {}
        emitted_signature_owner: dict[str, str] = {}
        duplicate_signature_skipped = 0

        def _try_emit(champ, h_bin, d_bin) -> tuple[bool, bool]:
            """Try to emit one instance from ``champ``.

            Returns (appended, advanced):
              appended — problem dict was appended to ``new_problems``.
              advanced — ``champ.execute`` was called (used to distinguish
                         transient seed failures from genuinely empty state).
            """
            nonlocal duplicate_signature_skipped
            pid = champ.program_id
            if emitted_per_program.get(pid, 0) >= self.instances_per_program:
                return False, False
            signature = self.map_elites.program_behavior_signature(champ)
            if signature:
                owner = emitted_signature_owner.get(signature)
                if owner is not None and owner != pid:
                    duplicate_signature_skipped += 1
                    return False, False
            seed = self._next_unused_seed(pid)
            inst = champ.execute(seed=seed)
            # Whether or not execute succeeds, mark the seed as consumed to
            # avoid infinite retries on a permanently-broken (program, seed).
            self.used_seeds.setdefault(pid, set()).add(seed)
            if inst is None:
                return False, True
            new_problems.append({
                "problem": inst.problem,
                "answer": inst.answer,
                "program_id": pid,
                "rq_score": champ.rq_score,
                "h_bin": h_bin,
                "d_bin": d_bin,
                "seed": seed,
            })
            emitted_per_program[pid] = emitted_per_program.get(pid, 0) + 1
            if signature:
                emitted_signature_owner.setdefault(signature, pid)
            return True, True

        if self.training_selection_mode == "uniform":
            # Legacy: frontier champions, clamped by instances_per_program and
            # training_budget.
            champions = self.map_elites.get_all_champions()
            for champ in champions:
                if not self._is_frontier_champion(champ):
                    non_frontier_skipped += 1
                    continue
                h_bin = getattr(champ, "niche_h", None)
                d_bin = getattr(champ, "niche_div", None)
                while (
                    emitted_per_program.get(champ.program_id, 0)
                    < self.instances_per_program
                    and len(new_problems) < self.training_budget
                ):
                    appended, advanced = _try_emit(champ, h_bin, d_bin)
                    if not advanced:
                        break  # champion hit its per-refresh cap
                if len(new_problems) >= self.training_budget:
                    break
            new_problems = new_problems[: self.training_budget]
        else:
            # H-priority + D-uniform + strict anti-reuse, repeated sweeps.
            # progress: appended at least one valid problem this sweep.
            # advanced: consumed at least one seed this sweep (execute may have
            #   returned None). Lets us distinguish an empty archive from one
            #   sweep of transient seed failures — see MAX_FAILED_SWEEPS.
            grid = self.map_elites.grid
            n_h = self.map_elites.n_h_bins
            n_d = self.map_elites.n_div_bins
            MAX_FAILED_SWEEPS = 2
            failed_sweeps = 0

            while len(new_problems) < self.training_budget:
                sweep += 1
                progress = False
                advanced = False
                for h_bin in range(n_h - 1, -1, -1):  # high H first
                    d_order = list(range(n_d))
                    random.shuffle(d_order)
                    for d_bin in d_order:
                        if len(new_problems) >= self.training_budget:
                            break
                        niche = grid.get((h_bin, d_bin))
                        if niche is None or niche.champion is None:
                            continue
                        champ = niche.champion
                        if not self._is_frontier_champion(champ):
                            # Extreme-p champion — archive material only,
                            # not training data. Counted per-sweep-visit, so
                            # later sweeps may re-count if still non-frontier.
                            non_frontier_skipped += 1
                            continue
                        appended, step_advanced = _try_emit(champ, h_bin, d_bin)
                        if appended:
                            progress = True
                        if step_advanced:
                            advanced = True
                    else:
                        continue
                    break  # budget reached in inner loop, propagate

                if progress:
                    failed_sweeps = 0
                    continue

                failed_sweeps += 1
                if not advanced:
                    # Normal outcome when frontier_champions ×
                    # instances_per_program < training_budget (small archive):
                    # every frontier champion hit its per-refresh emission cap
                    # this sweep. Info-level.
                    logger.info(
                        f"[Evolution] Training budget under-filled (expected "
                        f"for small archive): {len(new_problems)}/"
                        f"{self.training_budget} after {sweep} sweeps; "
                        f"all frontier champions at instances_per_program="
                        f"{self.instances_per_program} cap"
                    )
                    break
                if failed_sweeps >= MAX_FAILED_SWEEPS:
                    # Seeds were attempted but execute() returned None — a real
                    # problem (exec-failing programs survived). Warning-level.
                    logger.warning(
                        f"[Evolution] Training budget under-filled with exec "
                        f"failures: {len(new_problems)}/{self.training_budget} "
                        f"after {sweep} sweeps; no valid instance appended in "
                        f"last {MAX_FAILED_SWEEPS} sweeps (seeds consumed but "
                        f"all execute() returned None)"
                    )
                    break

        # ALWAYS update + rebuild, even if new_problems is empty. Keeping stale
        # problems from now-evicted champions in dynamic_dataset would contradict
        # the self-invalidating archive. Empty dataset is loud-logged below.
        #
        # Note: MapElitesDynamicDataset.__len__ returns max(n_problems, 1) for
        # robustness against edge-case __getitem__ lookups, so len(dataset) can
        # never be 0. Use .snapshot() for the TRUE problem count.
        self.dynamic_dataset.update(new_problems)
        self._rebuild_dataloader()
        new_size = len(self.dynamic_dataset.snapshot())
        batch = self.config.data.rollout_batch_size
        steps_per_outer_iteration = new_size // batch
        dropped = new_size - steps_per_outer_iteration * batch

        # Archive-wide frontier accounting (reporting only).
        all_champs = self.map_elites.get_all_champions()
        frontier_champs = sum(1 for c in all_champs if self._is_frontier_champion(c))
        self._last_refresh_stats = {
            "archive_champions": len(all_champs),
            "frontier_champions": frontier_champs,
            "non_frontier_skipped_visits": non_frontier_skipped,
            "duplicate_signature_skipped_visits": duplicate_signature_skipped,
        }

        # Persist the actual training problem list alongside the grid snapshot.
        # Without this the dataset that actually reached the Solver is not
        # recoverable after the run, making "why did dataset_size become X?"
        # impossible to answer from logs alone.
        self._save_dataset_snapshot(new_problems, frontier_champs, len(all_champs))

        logger.info(
            f"[Evolution] Dataset refreshed ({self.training_selection_mode}): "
            f"{old_size} → {new_size} problems, budget={self.training_budget}, "
            f"sweeps={sweep}, steps_per_outer_iteration={steps_per_outer_iteration}, "
            f"archive={len(all_champs)} champs "
            f"(frontier={frontier_champs}), "
            f"non_frontier_skipped_visits={non_frontier_skipped}, "
            f"duplicate_signature_skipped_visits={duplicate_signature_skipped}, "
            f"dataloader rebuilt"
        )
        if new_size == 0:
            logger.error(
                f"[Evolution] Dataset is EMPTY after refresh — no frontier "
                f"champion produced a valid instance. Training will yield 0 "
                f"steps this outer iteration. Likely causes: all champions are extreme-p "
                f"(p=0 or p=1), or frontier champions all exec-failed, or the "
                f"archive itself is empty. Investigate reeval / mutation logs."
            )
        elif dropped > 0:
            logger.warning(
                f"[Evolution] drop_last=True will drop {dropped} examples per "
                f"outer iteration (budget={self.training_budget} not divisible by "
                f"rollout_batch_size={batch})"
            )
        if 0 < new_size < batch:
            logger.warning(
                f"[Evolution] dataset_size={new_size} < rollout_batch_size="
                f"{batch}; drop_last=True will yield 0 training steps this outer iteration"
            )

    def _rebuild_dataloader(self):
        """dynamic_dataset 변경 후 train_dataloader를 재구성.

        현재 outer iteration의 iterator는 이미 생성되어 있으므로 즉시 반영되지 않지만,
        num_workers=0 + __getitem__의 thread-safe update 덕분에
        데이터 내용 자체는 즉시 반영된다.
        다음 outer iteration부터는 새 dataloader의 sampler 범위가 적용된다.
        """
        # shuffle MUST be honored — _refresh_dataset builds new_problems in
        # H-priority order, so without shuffle, every outer iteration's early batches
        # would be high-H biased relative to the intended training distribution.
        self.train_dataloader = StatefulDataLoader(
            dataset=self.dynamic_dataset,
            batch_size=self.config.data.rollout_batch_size,
            num_workers=0,
            shuffle=self.config.data.shuffle,
            drop_last=True,
            collate_fn=verl_collate_fn,
        )
