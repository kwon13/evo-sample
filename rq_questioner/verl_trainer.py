"""
RQ-Evolve Trainer: veRL RayPPOTrainer + MAP-Elites evolution.

Method-aligned epoch flow (method.tex §coevolution):
  매 Solver epoch 시작에서 _pre_epoch_hook()이 evolution을 수행.

  Epoch 0 .. N-1:
    0. _pre_epoch_hook(epoch_idx) → _evolution_step():
       a. Champion re-evaluation: filled niche의 champion을 현재 Solver
          기준으로 R_Q 재측정 (self-invalidating archive)
       b. Mutation rounds × max_rounds:
          - MAP-Elites parent 샘플링 → LLM mutation → multi-seed 자가 검증
          - entropy probe → H pre-filter → G-1 expand rollout → R_Q scoring
          - grid try_insert / evict
       c. _refresh_dataset(): H-priority + D-uniform + strict anti-reuse로
          epoch dataset 재조립 → dataloader rebuild
    1. Solver epoch: refresh된 dataset으로 rollout → reward → REINFORCE++
    2. val_freq 주기로 validation (수학 능력 추적)

  step-based _pre_actor_update_hook trigger는 제거됨.
"""

import re
import sys
import uuid
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
from .rq_score import compute_rq_full, h_prefilter
from .verl_dataset import MapElitesDynamicDataset

from prompts import (
    MUTATE_DEPTH, MUTATE_BREADTH, MUTATE_CROSSOVER,
    build_score_feedback, build_few_shot_examples, build_execution_feedback,
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

    instances = []
    for s in range(n_seeds):
        inst = program.execute(seed=s, timeout=5.0)
        if inst is None:
            return None  # 하나라도 실행 실패 → 거부
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

    return instances[0] if instances else None






def _extract_boxed(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def _normalize(s: str) -> str:
    s = s.strip().lower()
    for o, c in [("{", "}"), ("[", "]"), ("(", ")")]:
        if s.startswith(o) and s.endswith(c):
            s = s[1:-1].strip()
    return " ".join(s.rstrip(".").split())


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


def _extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = m.group(1).strip() if m else text.strip()
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith(("def generate", "import", "from")):
            code = "\n".join(lines[i:])
            break
    return code if "def generate" in code else None


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
    RayPPOTrainer + MAP-Elites evolution (Method-aligned epoch flow).

    fit() 흐름:
      epoch 0 .. N-1:
        1. _pre_epoch_hook(epoch_idx) → _evolution_step():
             a. Champion re-evaluation (self-invalidating archive)
             b. Mutation rounds × max_rounds (probe → H prefilter → expand G-1)
             c. _refresh_dataset() → dataloader rebuild
        2. 이 epoch 의 batch 들을 순회 → Solver rollout → REINFORCE++ update
        3. val_freq 마다 validation

    로깅:
      - evo/* : grid 상태, champion 분포, accept rate, reeval stats (Tracker 경유)
      - val/*  : solver accuracy on val set (Tracker 경유)
      - evolution_logs/*.json : grid 스냅샷 (파일)
    """

    def __init__(
        self,
        *args,
        map_elites: MAPElitesGrid,
        dynamic_dataset: MapElitesDynamicDataset,
        candidates_per_evo: int = 8,
        max_rounds: int = 8,
        num_rollouts: int = 16,
        instances_per_program: int = 3,
        in_depth_ratio: float = 0.5,
        crossover_ratio: float = 0.2,
        h_threshold: float = 0.1,
        # --- Champion re-evaluation (Method-aligned self-invalidating archive) ---
        # None (default) — re-evaluate every occupied champion each evolution step.
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
        # --- Epoch-start evolution hook ---
        evolve_before_train: bool = True,
        skip_initial_evolution_on_resume: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.map_elites = map_elites
        self.dynamic_dataset = dynamic_dataset
        self.candidates_per_evo = candidates_per_evo
        self.max_rounds = max_rounds
        self.num_rollouts = num_rollouts
        self.instances_per_program = instances_per_program
        self.in_depth_ratio = in_depth_ratio
        self.crossover_ratio = crossover_ratio
        self.h_threshold = h_threshold
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
        # epoch-hook flags (resume detection uses load_checkpoint_path presence)
        self.evolve_before_train = bool(evolve_before_train)
        self.skip_initial_evolution_on_resume = bool(skip_initial_evolution_on_resume)
        self._is_resume = bool(
            getattr(self.config.trainer, "load_checkpoint_path", None)
        )
        self.used_seeds: dict[str, set[int]] = {}

    # ------------------------------------------------------------------
    # Hook: called per mini-batch update
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # DP-safe worker invocation — pads odd batches to world_size then unpads.
    # evolution/reeval produce arbitrary batch sizes (n_children, retained_indices,
    # extra_n × retained) that may be odd under 2-GPU DP, tripping
    # DataProto.chunk(world_size) assertions. Mirrors ray_trainer._validate().
    # ------------------------------------------------------------------

    def _generate_sequences_dp_safe(self, batch: DataProto) -> DataProto:
        world_size = self.actor_rollout_wg.world_size
        batch, pad = pad_dataproto_to_divisor(batch, world_size)
        out = self.actor_rollout_wg.generate_sequences(batch)
        return unpad_dataproto(out, pad)

    def _compute_log_probs_dp_safe(
        self, batch: DataProto, calculate_entropy: bool, temperature: float,
    ) -> DataProto:
        world_size = self.actor_rollout_wg.world_size
        batch, pad = pad_dataproto_to_divisor(batch, world_size)
        batch.meta_info["calculate_entropy"] = calculate_entropy
        batch.meta_info["temperature"] = temperature
        out = self.actor_rollout_wg.compute_log_probs(batch)
        return unpad_dataproto(out, pad)

    def _pre_epoch_hook(self, epoch_idx: int) -> dict | None:
        """
        Method-aligned epoch-start hook.

        fit()의 매 epoch 시작에서 호출되어 Questioner evolution 을 돌린다.
        _evolution_step() 내부에서 champion re-evaluation → mutation rounds →
        dataset refresh 가 일어나므로, epoch 의 첫 batch 는 refresh 된
        archive-selected dataset 을 사용한다.

        Epoch 0 에서만 추가 분기:
          - evolve_before_train=false → skip
          - resume + skip_initial_evolution_on_resume=true → skip
            (checkpoint 는 이미 fit() 시작부에서 load 되었으므로, 원한다면
             resume 에서도 evolution 을 돌리는 것이 safe — skip 은 옵션)
        """
        if epoch_idx == 0:
            if not self.evolve_before_train:
                logger.info("[Evolution] epoch 0: skipped (evolve_before_train=false)")
                return None
            if self._is_resume and self.skip_initial_evolution_on_resume:
                logger.info(
                    "[Evolution] epoch 0: skipped "
                    "(resume + skip_initial_evolution_on_resume=true)"
                )
                return None

        logger.info(f"[Evolution] Epoch {epoch_idx} start")
        evo_metrics = self._evolution_step()

        # 핵심 지표 요약 출력 (console에서 한눈에 파악)
        print(
            f"\n{'='*60}\n"
            f"[Evolution Summary] epoch={epoch_idx}\n"
            f"  Reeval:   {evo_metrics['reeval_count']} updated, "
            f"{evo_metrics['reeval_evicted']} evicted "
            f"(low_h={evo_metrics.get('reeval_low_h_evicted', 0)}, "
            f"exec_fail={evo_metrics.get('reeval_exec_fail_evicted', 0)}), "
            f"{evo_metrics['reeval_bin_shifted']} bin-shifted "
            f"(ΔR_Q={evo_metrics['reeval_rq_delta_mean']:+.4f})\n"
            f"  Mutation: {evo_metrics['attempted']} attempted, "
            f"{evo_metrics['inserted']} inserted "
            f"({evo_metrics['accept_rate']:.0%} accept)\n"
            f"  Grid: {evo_metrics['grid_champions']}/{evo_metrics['total_niches']} niches filled "
            f"({evo_metrics['grid_coverage']:.0%} coverage), "
            f"H2+={evo_metrics['hard_champions']}\n"
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

    # ------------------------------------------------------------------
    # Evolution step (driver-side, CPU)
    # ------------------------------------------------------------------

    def _evolution_step(self) -> dict:
        """
        Fixed-budget evolution (FunSearch 스타일).
        매 step 고정 라운드 수(max_rounds)만큼 탐색. 조기 종료 없음.

        순서:
          1. (옵션 A) 기존 champion 재평가 — stale R_Q 갱신
          2. Mutation 라운드 max_rounds 회
          3. Dataset refresh
        """
        logger.info(
            f"[Evolution] step at global_step={getattr(self, 'global_step', '?')} "
            f"(max_rounds={self.max_rounds}, candidates_per_round={self.candidates_per_evo})"
        )

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
                f"(low_h={reeval_metrics.get('low_h_evicted', 0)}, "
                f"exec_fail={reeval_metrics.get('exec_fail_evicted', 0)}), "
                f"bin_shifted={reeval_metrics['bin_shifted']}, "
                f"ΔR_Q mean={reeval_metrics['rq_delta_mean']:+.4f}"
            )

        total_attempted = 0
        total_inserted = 0

        for round_num in range(1, self.max_rounds + 1):
            hard = self.map_elites.count_hard_champions(min_h_bin=2)
            logger.info(
                f"[Evolution] Round {round_num}/{self.max_rounds}: "
                f"{self.candidates_per_evo} candidates (H2+={hard})"
            )

            attempted, inserted = self._evolution_round(self.candidates_per_evo)
            total_attempted += attempted
            total_inserted += inserted

        self._refresh_dataset()

        stats = self.map_elites.stats()

        # Champion 분포 통계 (p_hat, H, R_Q)
        champions = self.map_elites.get_all_champions()
        p_hats = [getattr(c, "p_hat", 0.0) for c in champions if c.rq_score]
        h_scores = [getattr(c, "h_score", 0.0) for c in champions if c.rq_score]

        # Frontier-status breakdown (archive material vs. training-eligible).
        # frontier_status is set by _evolution_round (new candidates) and
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
            # Evolution round stats
            "attempted": total_attempted,
            "inserted": total_inserted,
            "accept_rate": total_inserted / max(total_attempted, 1),
            "rounds": round_num,
            # Grid state
            "grid_coverage": stats["coverage"],
            "grid_mean_rq": stats["mean_rq"],
            "grid_max_rq": stats["max_rq"],
            "grid_min_rq": stats["min_rq"],
            "grid_champions": stats["num_champions"],
            "hard_champions": stats["hard_champions"],
            "total_niches": stats["total_niches"],
            "total_insertions": stats["total_insertions"],
            "total_replacements": stats["total_replacements"],
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
            # Champion re-evaluation (self-invalidating archive)
            "reeval_count": reeval_metrics["reevaluated"],
            "reeval_evicted": reeval_metrics["evicted"],
            "reeval_low_h_evicted": reeval_metrics.get("low_h_evicted", 0),
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
        }

        # Grid 스냅샷 JSON 저장 (시각화는 별도 스크립트로)
        self._save_evolution_snapshot(result)

        return result

    def _save_evolution_snapshot(self, evo_metrics: dict):
        """매 evolution step마다 grid 상태 + 챔피언 정보를 JSON으로 저장."""
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
                    "problem": inst.problem if inst else "",
                    "answer": inst.answer if inst else "",
                    "source_code": champ.source_code,
                })
            grid_snapshot.append(cell)

        snapshot = {
            "global_step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": evo_metrics,
            "grid": grid_snapshot,
            # Generic D-axis labels. MAPElitesGrid now uses PCA projection on
            # problem embeddings (fit_diversity_axis) so bins don't map to
            # specific seeds anymore.
            "seed_labels": {
                str(d): f"D{d}" for d in range(self.map_elites.n_div_bins)
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

          이 메서드는 매 evolution step 앞에서:
            - age-weighted 로 champion 샘플링 (오래 재평가 안 된 것 우선)
            - fresh seed 로 instance 재생성
            - 현재 solver 로 p_hat / H 측정 → R_Q 갱신
            - p_hat 극단인 champion 은 niche 에서 evict → 재탐색 유도
            - H 재측정으로 bin 경계를 넘으면 rebin_champion() 으로 이동

        Returns:
            metrics dict — reevaluated / evicted / bin_shifted /
                           rq_delta_{mean,min,max}
        """
        # All counters initialized at function entry so early-return paths
        # (no champions, all execute-failing) carry the evictions they did.
        evicted = 0
        exec_fail_evicted = 0
        low_h_evicted = 0
        extreme_p_kept = 0
        bin_shifted = 0
        reevaluated = 0
        rq_deltas: list[float] = []

        def _metrics() -> dict:
            return {
                "reevaluated": reevaluated,
                "evicted": evicted,
                "low_h_evicted": low_h_evicted,
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

        # Solver prompt 구성 (기존 _evolution_round 와 동일)
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

        # ---- 3. Probe generate (n=1) + full-vocab entropy -------------
        # amortization 구조: probe 1회로 H̄ 측정 → H prefilter → 살아남은
        # 후보에만 G-1 expand rollout. probe response는 첫 rollout으로 재사용.
        n_pairs = len(pairs)
        pair_answers = [inst.answer for _, inst, _ in pairs]

        probe_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=solver_texts,
            answers=pair_answers,
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
            n_repeat=1,
        )
        try:
            probe_output = self._generate_sequences_dp_safe(probe_batch)
        except Exception as e:
            logger.warning(f"[Reeval] probe rollout failed: {e}")
            return _metrics()

        probe_resp_ids = probe_output.batch.get("responses")
        response_mask = probe_output.batch.get("response_mask")
        if probe_resp_ids is None or response_mask is None:
            return _metrics()

        probe_flags: list[bool] = []
        for ci in range(n_pairs):
            decoded = self.tokenizer.decode(
                probe_resp_ids[ci].tolist(), skip_special_tokens=True,
            )
            pred = _extract_boxed(decoded)
            probe_flags.append(_answers_match(pred, pair_answers[ci]) if pred else False)

        try:
            entropy_output = self._compute_log_probs_dp_safe(
                probe_output, calculate_entropy=True, temperature=temperature,
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

        mask_f = response_mask.float()
        h_tensor = (entropies * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
        h_per_pair: list[float] = h_tensor.cpu().tolist()

        # ---- 4. Counters (evicted / exec_fail_evicted already initialized) --
        # frontier_p_hat_range defines the training-data frontier band; it is
        # NOT a reeval eviction gate anymore. Extreme-p champions stay in the
        # archive as mutation material and are excluded from training via
        # _refresh_dataset(); their R_Q naturally collapses to 0.
        low, high = self.frontier_p_hat_range
        extreme_p_kept = 0

        # ---- 5. H prefilter — low-H champions are evicted --------------
        # Method's self-invalidating archive: a champion whose current H̄ fell
        # below τ_H no longer satisfies R_Q's learnability criterion. Keeping
        # it would preserve stale high R_Q in the grid. Evict instead of skip.
        retained_indices: list[int] = []
        for ci in range(n_pairs):
            champ, inst, _ = pairs[ci]
            h_bar = h_per_pair[ci]
            if not h_prefilter(h_bar, self.h_threshold):
                champ.last_reeval_step = current_step
                if self.map_elites.evict_champion(champ):
                    evicted += 1
                    low_h_evicted += 1
                    logger.info(
                        f"[Reeval] EVICT low-H ({champ.niche_h},{champ.niche_div}) "
                        f"H={h_bar:.3f} < τ_H={self.h_threshold:.3f}, "
                        f"id={champ.program_id}"
                    )
                continue
            retained_indices.append(ci)

        if low_h_evicted:
            logger.info(
                f"[Reeval] low-H evicted {low_h_evicted}/{n_pairs}; "
                f"saved {low_h_evicted * (self.num_rollouts - 1)} expand rollouts"
            )

        # ---- 6. Expand rollout for retained champions -----------------
        extra_flags_per_ci: dict[int, list[bool]] = {ci: [] for ci in retained_indices}
        extra_n = self.num_rollouts - 1
        if retained_indices and extra_n > 0:
            retained_texts = [solver_texts[ci] for ci in retained_indices]
            retained_answers = [pair_answers[ci] for ci in retained_indices]
            extra_batch = _make_gen_batch(
                tokenizer=self.tokenizer,
                prompts_text=retained_texts,
                answers=retained_answers,
                temperature=temperature,
                eos_token_id=eos_id, pad_token_id=pad_id,
                max_prompt_length=self.config.data.max_prompt_length,
                n_repeat=extra_n,
            )
            try:
                extra_output = self._generate_sequences_dp_safe(extra_batch)
            except Exception as e:
                logger.warning(f"[Reeval] expand rollout failed: {e}")
                return _metrics()

            extra_resp_ids = extra_output.batch.get("responses")
            if extra_resp_ids is None:
                return _metrics()

            for local_idx, ci in enumerate(retained_indices):
                ans = pair_answers[ci]
                for ri in range(extra_n):
                    idx = local_idx * extra_n + ri
                    decoded = self.tokenizer.decode(
                        extra_resp_ids[idx].tolist(), skip_special_tokens=True,
                    )
                    pred = _extract_boxed(decoded)
                    extra_flags_per_ci[ci].append(
                        _answers_match(pred, ans) if pred else False
                    )

        # ---- 7. Update / rebin — NEVER evict on extreme-p anymore ------
        # p=0 / p=1 champions are kept as archive material and will be
        # filtered out of Solver training data by _refresh_dataset().
        # Their R_Q naturally drops to 0 (since p(1-p)=0), so they cannot
        # displace positive-R_Q candidates at try_insert time.
        for ci in retained_indices:
            champ, inst, rq_before = pairs[ci]
            h_bar = h_per_pair[ci]
            flags = [probe_flags[ci]] + extra_flags_per_ci[ci]
            new_p_hat = sum(flags) / len(flags) if flags else 0.0
            new_rq = new_p_hat * (1.0 - new_p_hat) * h_bar

            champ.p_hat = new_p_hat
            champ.h_score = h_bar
            champ.rq_score = new_rq
            champ.fitness = new_rq
            champ.last_reeval_step = current_step
            champ.metadata["frontier_status"] = (
                "too_hard" if new_p_hat <= low
                else "too_easy" if new_p_hat >= high
                else "frontier"
            )
            if new_p_hat <= low or new_p_hat >= high:
                extreme_p_kept += 1

            moved = self.map_elites.rebin_champion(
                program=champ, new_h_value=h_bar, problem_text=inst.problem,
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

    def _evolution_round(self, batch_size: int) -> tuple[int, int]:
        """
        단일 라운드: batch_size개 candidate mutation → rollout → entropy → grid insert.
        _refresh_dataset()는 호출하지 않음 (caller가 최종 1회 호출).

        Returns: (attempted, inserted)
        """
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        temperature = self.config.worker.rollout.temperature

        # ================================================================
        # Phase 1: Mutation — 연산자 선택 + batch generate
        # ================================================================

        few_shot = build_few_shot_examples(self.map_elites, top_k=3)

        mutation_prompts = []  # (prompt_text, op, parent, parent_b)
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
                    mutation_prompts.append((prompt_text, op, pa, pb))
                    continue

            parent = self.map_elites.sample_parent() if op != "crossover" else pa
            if parent is None:
                continue

            score_fb = build_score_feedback(parent)
            exec_fb = build_execution_feedback(parent)
            tmpl = MUTATE_DEPTH if op == "in_depth" else MUTATE_BREADTH
            prompt_text = tmpl.format(
                code=parent.source_code,
                score_feedback=score_fb,
                exec_feedback=exec_fb,
                few_shot=few_shot,
            )
            mutation_prompts.append((prompt_text, op, parent, None))

        if not mutation_prompts:
            return 0, 0

        mut_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=[pt for pt, _, _, _ in mutation_prompts],
            answers=[""] * len(mutation_prompts),
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
        )

        try:
            mut_output = self._generate_sequences_dp_safe(mut_batch)
        except Exception as e:
            logger.warning(f"[Evolution] batch mutation failed: {e}")
            return 0, 0

        resp_ids = mut_output.batch.get("responses")
        if resp_ids is None:
            return 0, 0

        # Decode + execute (CPU)
        children = []  # (child, inst, op)
        for i, (_, op, parent, parent_b) in enumerate(mutation_prompts):
            code_text = self.tokenizer.decode(
                resp_ids[i].tolist(), skip_special_tokens=True
            )
            source_code = _extract_code(code_text)
            if not source_code:
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
                continue
            children.append((child, inst, op))

        if not children:
            return 0, 0

        # ================================================================
        # Phase 2-4: entropy probe → H prefilter → expanded rollout → R_Q
        # ================================================================
        # method.tex의 amortization 구조를 그대로 실현:
        #   (a) probe 1회 generate + actor forward로 full-vocab H̄ 계산
        #   (b) H̄ < τ_H 후보는 G rollout 자체를 skip (비용 절감)
        #   (c) 살아남은 후보만 G-1번 추가 rollout → probe와 합쳐 G개 flag
        solver_texts = []
        for _, inst, _ in children:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": inst.problem}]
            if self.tokenizer.chat_template:
                solver_texts.append(self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
            else:
                solver_texts.append(f"system: {SYSTEM_PROMPT}\nuser: {inst.problem}")

        n_children = len(children)
        inst_answers = [inst.answer for _, inst, _ in children]

        # ---- (a-1) Probe generate: n_repeat=1 at rollout temperature ----
        probe_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=solver_texts,
            answers=inst_answers,
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
            n_repeat=1,
        )
        try:
            probe_output = self._generate_sequences_dp_safe(probe_batch)
        except Exception as e:
            logger.warning(f"[Evolution] probe rollout failed: {e}")
            return 0, 0

        probe_resp_ids = probe_output.batch.get("responses")
        response_mask = probe_output.batch.get("response_mask")
        if probe_resp_ids is None or response_mask is None:
            return 0, 0

        # Decode probe flag per child (reused as the 1st of G rollouts).
        probe_flags: list[bool] = []
        for ci in range(n_children):
            decoded = self.tokenizer.decode(
                probe_resp_ids[ci].tolist(), skip_special_tokens=True
            )
            pred = _extract_boxed(decoded)
            probe_flags.append(_answers_match(pred, inst_answers[ci]) if pred else False)

        # ---- (a-2) Full-vocab entropy via actor forward pass -----------
        try:
            entropy_output = self._compute_log_probs_dp_safe(
                probe_output, calculate_entropy=True, temperature=temperature,
            )
        except Exception as e:
            logger.warning(f"[Evolution] entropy forward failed: {e}")
            return 0, 0

        entropies = entropy_output.batch.get("entropies")
        if entropies is None:
            logger.warning("[Evolution] actor did not return entropies")
            return 0, 0
        assert entropies.shape == response_mask.shape, (
            f"[Evolution] entropies {tuple(entropies.shape)} != response_mask "
            f"{tuple(response_mask.shape)}"
        )

        mask_f = response_mask.float()
        h_tensor = (entropies * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
        h_per_child: list[float] = h_tensor.cpu().tolist()

        # ---- (b) H prefilter: retained set only proceeds to expand -----
        retained_indices: list[int] = []
        skipped_h = 0
        for ci in range(n_children):
            h_bar = h_per_child[ci]
            if not h_prefilter(h_bar, self.h_threshold):
                skipped_h += 1
                logger.debug(
                    f"[Evolution] H={h_bar:.3f} < {self.h_threshold}, "
                    f"skip rollout (amortized)"
                )
                continue
            retained_indices.append(ci)

        if skipped_h:
            logger.info(
                f"[Evolution] H prefilter: {skipped_h}/{n_children} candidates "
                f"skipped before rollout (saved ~{skipped_h * (self.num_rollouts - 1)} generations)"
            )

        # ---- (c) Expand rollout: G-1 more per retained candidate -------
        extra_flags_per_ci: dict[int, list[bool]] = {ci: [] for ci in retained_indices}
        extra_n = self.num_rollouts - 1
        if retained_indices and extra_n > 0:
            retained_texts = [solver_texts[ci] for ci in retained_indices]
            retained_answers = [inst_answers[ci] for ci in retained_indices]
            extra_batch = _make_gen_batch(
                tokenizer=self.tokenizer,
                prompts_text=retained_texts,
                answers=retained_answers,
                temperature=temperature,
                eos_token_id=eos_id, pad_token_id=pad_id,
                max_prompt_length=self.config.data.max_prompt_length,
                n_repeat=extra_n,
            )
            try:
                extra_output = self._generate_sequences_dp_safe(extra_batch)
            except Exception as e:
                logger.warning(f"[Evolution] expand rollout failed: {e}")
                return n_children, 0

            extra_resp_ids = extra_output.batch.get("responses")
            if extra_resp_ids is None:
                return n_children, 0

            for local_idx, ci in enumerate(retained_indices):
                ans = inst_answers[ci]
                for ri in range(extra_n):
                    idx = local_idx * extra_n + ri
                    decoded = self.tokenizer.decode(
                        extra_resp_ids[idx].tolist(), skip_special_tokens=True,
                    )
                    pred = _extract_boxed(decoded)
                    extra_flags_per_ci[ci].append(
                        _answers_match(pred, ans) if pred else False
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
        for ci in retained_indices:
            child, inst, op = children[ci]
            h_bar = h_per_child[ci]
            flags = [probe_flags[ci]] + extra_flags_per_ci[ci]
            p_hat = sum(flags) / len(flags) if flags else 0.0

            rq_result = compute_rq_full(flags, h_bar)
            child.p_hat = p_hat
            child.h_score = h_bar
            child.rq_score = rq_result.rq_score
            child.fitness = rq_result.rq_score
            child.metadata["frontier_status"] = (
                "too_hard" if p_hat <= f_low
                else "too_easy" if p_hat >= f_high
                else "frontier"
            )

            was_inserted = self.map_elites.try_insert(
                program=child, h_value=h_bar,
                problem_text=inst.problem, rq_score=rq_result.rq_score,
            )
            if was_inserted:
                inserted += 1
                logger.info(
                    f"[Evolution] Inserted ({op}, "
                    f"{child.metadata['frontier_status']}): p={p_hat:.3f} "
                    f"H={h_bar:.3f} R_Q={rq_result.rq_score:.4f}"
                )

        return attempted, inserted

    # ------------------------------------------------------------------
    # Dataset refresh from MAP-Elites champions
    # ------------------------------------------------------------------

    def _next_unused_seed(self, program_id: str) -> int | None:
        """Smallest non-negative integer not yet consumed by this program.

        When ``strict_anti_reuse`` is off, any seed up to ``instances_per_program``
        may repeat across epochs; this method then simply cycles through the
        fixed range. When strict, the counter monotonically increases, so each
        (program_id, seed) pair is used at most once across the whole run.
        """
        used = self.used_seeds.setdefault(program_id, set())
        if not self.strict_anti_reuse:
            for s in range(self.instances_per_program):
                if s not in used:
                    return s
            return None
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
        frontier so the very first epoch's dataset isn't silently emptied.
        """
        p = getattr(champ, "p_hat", None)
        if p is None:
            return True
        low, high = self.frontier_p_hat_range
        return float(low) < float(p) < float(high)

    def _refresh_dataset(self):
        """MAP-Elites 챔피언 → dynamic_dataset 교체 → dataloader 재구성.

        Selection rule (``h_priority_d_uniform``):
          Repeated H×D sweeps until ``training_budget`` is reached. Each sweep
          processes every filled niche at most once (≤1 new problem per niche),
          using :meth:`_next_unused_seed` for strict anti-reuse. Across sweeps,
          the same champion may be revisited with its next unused seed until
          the budget is hit or no sweep makes any progress.

          Only frontier champions (0 < p_hat < 1) contribute to training data —
          extreme-p champions stay in the archive as mutation material.

        The legacy ``uniform`` mode iterates every frontier champion with seeds
        ``0..instances_per_program`` then clamps to ``training_budget``.
        """
        old_size = len(self.dynamic_dataset.snapshot())
        new_problems: list[dict] = []
        sweep = 0
        non_frontier_skipped = 0  # reporting-only counter

        if self.training_selection_mode == "uniform":
            # Legacy: frontier champions × fixed seed range, clamped to budget.
            champions = self.map_elites.get_all_champions()
            for champ in champions:
                if not self._is_frontier_champion(champ):
                    non_frontier_skipped += 1
                    continue
                for seed in range(self.instances_per_program):
                    inst = champ.execute(seed=seed)
                    if inst:
                        new_problems.append({
                            "problem": inst.problem,
                            "answer": inst.answer,
                            "program_id": champ.program_id,
                            "rq_score": champ.rq_score,
                        })
                        if len(new_problems) >= self.training_budget:
                            break
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
                        seed = self._next_unused_seed(champ.program_id)
                        if seed is None:
                            continue
                        advanced = True
                        self.used_seeds.setdefault(champ.program_id, set()).add(seed)
                        inst = champ.execute(seed=seed)
                        if inst is None:
                            continue
                        new_problems.append({
                            "problem": inst.problem,
                            "answer": inst.answer,
                            "program_id": champ.program_id,
                            "rq_score": champ.rq_score,
                            "h_bin": h_bin,
                            "d_bin": d_bin,
                            "seed": seed,
                        })
                        progress = True
                    else:
                        continue
                    break  # budget reached in inner loop, propagate

                if progress:
                    failed_sweeps = 0
                    continue

                failed_sweeps += 1
                if not advanced:
                    logger.warning(
                        f"[Evolution] Training budget under-filled: "
                        f"{len(new_problems)}/{self.training_budget} after "
                        f"{sweep} sweeps; no usable champion/seed made progress"
                    )
                    break
                if failed_sweeps >= MAX_FAILED_SWEEPS:
                    logger.warning(
                        f"[Evolution] Training budget under-filled: "
                        f"{len(new_problems)}/{self.training_budget} after "
                        f"{sweep} sweeps; no valid instance appended in last "
                        f"{MAX_FAILED_SWEEPS} sweeps (seeds consumed but all "
                        f"execute() returned None)"
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
        steps_per_epoch = new_size // batch
        dropped = new_size - steps_per_epoch * batch

        # Archive-wide frontier accounting (reporting only).
        all_champs = self.map_elites.get_all_champions()
        frontier_champs = sum(1 for c in all_champs if self._is_frontier_champion(c))
        self._last_refresh_stats = {
            "archive_champions": len(all_champs),
            "frontier_champions": frontier_champs,
            "non_frontier_skipped_visits": non_frontier_skipped,
        }

        logger.info(
            f"[Evolution] Dataset refreshed ({self.training_selection_mode}): "
            f"{old_size} → {new_size} problems, budget={self.training_budget}, "
            f"sweeps={sweep}, steps_per_epoch={steps_per_epoch}, "
            f"archive={len(all_champs)} champs "
            f"(frontier={frontier_champs}), "
            f"non_frontier_skipped_visits={non_frontier_skipped}, "
            f"dataloader rebuilt"
        )
        if new_size == 0:
            logger.error(
                f"[Evolution] Dataset is EMPTY after refresh — no frontier "
                f"champion produced a valid instance. Training will yield 0 "
                f"steps this epoch. Likely causes: all champions are extreme-p "
                f"(p=0 or p=1), or frontier champions all exec-failed, or the "
                f"archive itself is empty. Investigate reeval / mutation logs."
            )
        elif dropped > 0:
            logger.warning(
                f"[Evolution] drop_last=True will drop {dropped} examples per "
                f"epoch (budget={self.training_budget} not divisible by "
                f"rollout_batch_size={batch})"
            )
        if 0 < new_size < batch:
            logger.warning(
                f"[Evolution] dataset_size={new_size} < rollout_batch_size="
                f"{batch}; drop_last=True will yield 0 training steps this epoch"
            )

    def _rebuild_dataloader(self):
        """dynamic_dataset 변경 후 train_dataloader를 재구성.

        현재 epoch의 iterator는 이미 생성되어 있으므로 즉시 반영되지 않지만,
        num_workers=0 + __getitem__의 thread-safe update 덕분에
        데이터 내용 자체는 즉시 반영된다.
        다음 epoch부터는 새 dataloader의 sampler 범위가 적용된다.
        """
        # shuffle MUST be honored — _refresh_dataset builds new_problems in
        # H-priority order, so without shuffle, every epoch's early batches
        # would be high-H biased relative to the intended training distribution.
        self.train_dataloader = StatefulDataLoader(
            dataset=self.dynamic_dataset,
            batch_size=self.config.data.rollout_batch_size,
            num_workers=0,
            shuffle=self.config.data.shuffle,
            drop_last=True,
            collate_fn=verl_collate_fn,
        )
