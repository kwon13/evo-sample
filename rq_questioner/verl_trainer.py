"""
RQ-Evolve Trainer: veRL RayPPOTrainer + MAP-Elites evolution.

Online 학습 파이프라인:
  RayPPOTrainer.fit() 루프를 그대로 사용하며,
  _pre_actor_update_hook()으로 evolution을 주입.

  매 step:
    1. Solver 학습: dynamic_dataset → rollout → reward → REINFORCE++ update
    2. evolution_freq마다 진화:
       a. MAP-Elites에서 parent 샘플링 → LLM mutation → 문제 생성 프로그램 변이
       b. 자가 검증 (multi-seed 실행 + SymPy)
       c. H pre-filter → rollout(G회) → p_hat 추정
       d. R_Q = p(1-p) · H_bar 계산 → grid 갱신
       e. _refresh_dataset() → dataloader 재구성
    3. val_freq마다 validation (수학 능력 추적)
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

from verl.protocol import DataProto
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.dataset import collate_fn as verl_collate_fn

from .map_elites import MAPElitesGrid
from .program import ProblemProgram, ProblemInstance
from .rq_score import compute_rq_full, h_prefilter, p_hat_filter
from .verl_dataset import MapElitesDynamicDataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)

# ---------------------------------------------------------------------------
# Score-aware mutation prompts (FunSearch, Romera-Paredes et al. 2023)
# ---------------------------------------------------------------------------

_SINGLE_ANSWER_RULE = (
    "RULES:\n"
    "1. Function MUST be named `generate` and take a single `seed` argument\n"
    "2. MUST return (problem_text: str, answer: str)\n"
    "3. answer MUST be a SINGLE number or simple value (e.g. '42', '3.14', '7/3')\n"
    "   NOT ranges, NOT multiple values, NOT inequalities\n"
    "4. Compute answer FIRST, then build problem from it\n"
    "5. Use only standard library + math module\n"
    "6. Self-contained, no external dependencies\n\n"
    "Return ONLY the Python code."
)

_SCORE_FEEDBACK = (
    "\n=== PERFORMANCE OF CURRENT PROGRAM ===\n"
    "pass_rate (p_hat) = {p_hat:.2f}  (ideal: 0.50)\n"
    "entropy (H)       = {h_score:.2f}  (ideal: > 2.0)\n"
    "R_Q score          = {rq_score:.4f}\n"
    "DIAGNOSIS: {diagnosis}\n"
    "ACTION: {action}\n\n"
)

def _score_diagnosis(p_hat: float, h_score: float) -> tuple[str, str]:
    if p_hat > 0.7:
        diag = f"TOO EASY (solver gets {p_hat:.0%} correct)"
        action = "Make problems significantly harder: more steps, combined concepts, larger numbers"
    elif p_hat < 0.2:
        diag = f"TOO HARD (solver gets only {p_hat:.0%} correct)"
        action = "Make problems slightly easier: clearer wording, fewer steps"
    else:
        diag = f"Good difficulty (p_hat={p_hat:.2f})"
        action = "Keep similar difficulty but increase problem diversity and reasoning depth"
    if h_score < 0.5:
        diag += f"; LOW ENTROPY (H={h_score:.2f})"
        action += "; add ambiguity or multi-step reasoning"
    return diag, action

def _build_score_feedback(parent: ProblemProgram) -> str:
    p_hat = getattr(parent, "p_hat", 0.5)
    h_score = getattr(parent, "h_score", 1.0)
    rq_score = getattr(parent, "rq_score", 0.0)
    diag, action = _score_diagnosis(p_hat, h_score)
    return _SCORE_FEEDBACK.format(
        p_hat=p_hat, h_score=h_score, rq_score=rq_score,
        diagnosis=diag, action=action,
    )

IN_DEPTH_PROMPT = (
    "You are an expert mathematician and Python programmer.\n\n"
    "{few_shot_examples}"
    "Below is a Python function that generates natural-language math word problems "
    "using inverse construction (answer chosen first, problem built from it).\n\n"
    "{score_feedback}"
    "{execution_feedback}"
    "```python\n{source_code}\n```\n\n"
    "Modify this function to generate HARDER problems (AMC/AIME level). You may:\n"
    "- Add more reasoning steps or constraints\n"
    "- Combine multiple math concepts\n"
    "- Require multi-step logic (3+ steps)\n\n"
    + _SINGLE_ANSWER_RULE
)

IN_BREADTH_PROMPT = (
    "You are an expert mathematician and Python programmer.\n\n"
    "{few_shot_examples}"
    "Below is a Python function that generates math word problems:\n\n"
    "{score_feedback}"
    "{execution_feedback}"
    "```python\n{source_code}\n```\n\n"
    "Create a COMPLETELY DIFFERENT type of math word problem generator covering "
    "a different branch of mathematics.\n\n"
    + _SINGLE_ANSWER_RULE
)

CROSSOVER_PROMPT = (
    "You are an expert mathematician and Python programmer.\n\n"
    "{few_shot_examples}"
    "Below are TWO Python functions that generate different types of math problems. "
    "Combine ideas from BOTH to create a NEW hybrid problem generator that merges "
    "the mathematical concepts from both parents.\n\n"
    "Parent A (p_hat={p_hat_a:.2f}, H={h_a:.2f}):\n"
    "```python\n{source_a}\n```\n\n"
    "Parent B (p_hat={p_hat_b:.2f}, H={h_b:.2f}):\n"
    "```python\n{source_b}\n```\n\n"
    "Create a function that combines concepts from both parents.\n\n"
    + _SINGLE_ANSWER_RULE
)

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


def _build_few_shot_examples(grid: MAPElitesGrid, top_k: int = 3) -> str:
    """
    Grid에서 RQ 상위 top_k 챔피언의 코드를 few-shot 예시로 구성.
    FunSearch (Romera-Paredes et al., 2023) 스타일.
    """
    champions = grid.get_all_champions()
    if not champions:
        return ""
    ranked = sorted(champions, key=lambda c: -(c.rq_score or 0))[:top_k]

    parts = ["=== HIGH-QUALITY EXAMPLES (for reference) ==="]
    for i, champ in enumerate(ranked, 1):
        rq = getattr(champ, "rq_score", 0)
        p = getattr(champ, "p_hat", 0)
        h = getattr(champ, "h_score", 0)
        parts.append(
            f"\nExample {i} (RQ={rq:.3f}, p_hat={p:.2f}, H={h:.2f}):\n"
            f"```python\n{champ.source_code}\n```"
        )
    parts.append("\n=== END EXAMPLES ===\n")
    return "\n".join(parts)


def _build_execution_feedback(parent: ProblemProgram) -> str:
    """
    부모 프로그램의 실행 결과를 프롬프트에 포함.
    생성된 문제 예시 + solver 성능 정보를 보여줌.
    """
    inst = parent.execute(seed=0, timeout=5.0)
    if inst is None:
        return ""

    p_hat = getattr(parent, "p_hat", 0.5)
    h_score = getattr(parent, "h_score", 1.0)

    feedback = (
        "\n=== EXECUTION RESULT OF CURRENT PROGRAM ===\n"
        f"Generated problem: {inst.problem}\n"
        f"Expected answer: {inst.answer}\n"
        f"Solver pass rate: {p_hat:.0%}"
    )
    if p_hat > 0.7:
        feedback += " (TOO EASY — solver answers correctly too often)\n"
    elif p_hat < 0.2:
        feedback += " (TOO HARD — solver almost never gets it right)\n"
    else:
        feedback += " (GOOD difficulty range)\n"
    feedback += f"Solver entropy: {h_score:.2f}"
    if h_score < 1.0:
        feedback += " (LOW — solver is too confident, needs more ambiguity)\n"
    else:
        feedback += " (OK)\n"
    feedback += "=== END EXECUTION RESULT ===\n\n"
    return feedback


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
    RayPPOTrainer + MAP-Elites evolution.

    Online 학습 흐름:
      fit() 루프 (RayPPOTrainer 그대로 사용):
        매 step:
          1. dynamic_dataset에서 배치 → Solver rollout → reward → REINFORCE++ update
          2. evolution_freq마다 _pre_actor_update_hook() 트리거:
             Inner Loop (_evolution_step):
               mutation → verify → rollout(p_hat) → entropy(H) → R_Q 계산
               → MAP-Elites grid 갱신 → _refresh_dataset() → dataloader 재구성
          3. val_freq마다 validation (수학 능력 추적)

    로깅:
      - evo/* : grid 상태, champion 분포, accept rate (Tracker 경유)
      - val/*  : solver accuracy on val set (Tracker 경유)
      - evolution_logs/*.json : grid 스냅샷 (파일)
    """

    def __init__(
        self,
        *args,
        map_elites: MAPElitesGrid,
        dynamic_dataset: MapElitesDynamicDataset,
        evolution_freq: int = 50,
        evolution_pct: float | None = None,
        candidates_per_evo: int = 8,
        max_rounds: int = 8,
        num_rollouts: int = 16,
        instances_per_program: int = 3,
        in_depth_ratio: float = 0.5,
        crossover_ratio: float = 0.2,
        h_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.map_elites = map_elites
        self.dynamic_dataset = dynamic_dataset
        self.evolution_freq = evolution_freq
        self.evolution_pct = evolution_pct
        self.candidates_per_evo = candidates_per_evo
        self.max_rounds = max_rounds
        self.num_rollouts = num_rollouts
        self.instances_per_program = instances_per_program
        self.in_depth_ratio = in_depth_ratio
        self.crossover_ratio = crossover_ratio
        self.h_threshold = h_threshold
        self._computed_evolution_freq = None

    # ------------------------------------------------------------------
    # Hook: called per mini-batch update
    # ------------------------------------------------------------------

    def _get_evolution_freq(self) -> int:
        """evolution_pct가 설정되면 total_steps에서 freq를 계산, 아니면 고정값 사용."""
        if self._computed_evolution_freq is not None:
            return self._computed_evolution_freq

        if self.evolution_pct is not None:
            total = getattr(self.config.trainer, 'max_steps', None) or \
                    getattr(self.config.trainer, 'total_training_steps', None)
            if total is None or total <= 0:
                total = len(self.train_dataloader) * self.config.trainer.total_epochs
            self._computed_evolution_freq = max(1, int(total * self.evolution_pct))
            logger.info(
                f"[Evolution] freq = {self._computed_evolution_freq} "
                f"({self.evolution_pct:.0%} of {total} steps)"
            )
        else:
            self._computed_evolution_freq = self.evolution_freq
            logger.info(f"[Evolution] freq = {self._computed_evolution_freq} (fixed)")

        return self._computed_evolution_freq

    def _pre_actor_update_hook(self, global_step: int) -> dict | None:
        """
        verl 0.3.1 fit() 루프에서 update_actor 전에 호출되는 hook.
        evolution_freq마다 _evolution_step() 실행.
        """
        freq = self._get_evolution_freq()
        if global_step % freq == 0:
            logger.info(f"[Evolution] Triggered at step {global_step}")
            evo_metrics = self._evolution_step()

            # 핵심 지표 요약 출력 (console에서 한눈에 파악)
            print(
                f"\n{'='*60}\n"
                f"[Evolution Summary] step={global_step}\n"
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
        return None

    # ------------------------------------------------------------------
    # Evolution step (driver-side, CPU)
    # ------------------------------------------------------------------

    def _evolution_step(self) -> dict:
        """
        Fixed-budget evolution (FunSearch 스타일).
        매 step 고정 라운드 수(max_rounds)만큼 탐색. 조기 종료 없음.
        """
        logger.info(
            f"[Evolution] step at global_step={getattr(self, 'global_step', '?')} "
            f"(max_rounds={self.max_rounds}, candidates_per_round={self.candidates_per_evo})"
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
            # Dataset state
            "dataset_size": len(self.dynamic_dataset),
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
            "seed_labels": {
                str(d): sid for sid, d in self.map_elites._seed_to_div.items()
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

        # Few-shot 예시 (top-3 챔피언, FunSearch 스타일)
        few_shot = _build_few_shot_examples(self.map_elites, top_k=3)

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
                    prompt_text = CROSSOVER_PROMPT.format(
                        source_a=pa.source_code, source_b=pb.source_code,
                        p_hat_a=getattr(pa, "p_hat", 0.5),
                        h_a=getattr(pa, "h_score", 1.0),
                        p_hat_b=getattr(pb, "p_hat", 0.5),
                        h_b=getattr(pb, "h_score", 1.0),
                        few_shot_examples=few_shot,
                    )
                    mutation_prompts.append((prompt_text, op, pa, pb))
                    continue

            parent = self.map_elites.sample_parent() if op != "crossover" else pa
            if parent is None:
                continue

            score_fb = _build_score_feedback(parent)
            exec_fb = _build_execution_feedback(parent)
            tmpl = IN_DEPTH_PROMPT if op == "in_depth" else IN_BREADTH_PROMPT
            prompt_text = tmpl.format(
                source_code=parent.source_code,
                score_feedback=score_fb,
                execution_feedback=exec_fb,
                few_shot_examples=few_shot,
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
            mut_output = self.actor_rollout_wg.generate_sequences(mut_batch)
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
        # Phase 2: Batch rollout → p_hat
        # ================================================================
        # Solver prompts (tokenized)
        solver_texts = []
        for _, inst, _ in children:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": inst.problem}]
            if self.tokenizer.chat_template:
                solver_texts.append(self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
            else:
                solver_texts.append(f"system: {SYSTEM_PROMPT}\nuser: {inst.problem}")

        rollout_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=solver_texts,
            answers=[inst.answer for _, inst, _ in children],
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
            n_repeat=self.num_rollouts,
        )

        try:
            rollout_output = self.actor_rollout_wg.generate_sequences(rollout_batch)
        except Exception as e:
            logger.warning(f"[Evolution] batch rollout failed: {e}")
            return 0, 0

        resp_ids = rollout_output.batch.get("responses")
        if resp_ids is None:
            return 0, 0

        # Decode rollout results per child
        n_children = len(children)
        all_flags = []
        for ci in range(n_children):
            flags = []
            for ri in range(self.num_rollouts):
                idx = ci * self.num_rollouts + ri
                decoded = self.tokenizer.decode(
                    resp_ids[idx].tolist(), skip_special_tokens=True
                )
                pred = _extract_boxed(decoded)
                _, inst, _ = children[ci]
                flags.append(_answers_match(pred, inst.answer) if pred else False)
            all_flags.append(flags)

        # ================================================================
        # Phase 3: Batch entropy (logprobs=20, temperature=1.0으로 별도 생성)
        # Feasibility test의 VLLMRunner.entropy()와 동일한 방식
        # ================================================================
        # entropy 측정: logprobs=20, temperature=1.0, max_tokens=256
        # meta_info로 sampling params 전달 → generate_sequences 내부에서 자동 적용
        entropy_batch = _make_gen_batch(
            tokenizer=self.tokenizer,
            prompts_text=solver_texts,
            answers=[inst.answer for _, inst, _ in children],
            temperature=1.0,
            eos_token_id=eos_id, pad_token_id=pad_id,
            max_prompt_length=self.config.data.max_prompt_length,
            n_repeat=1,
        )
        entropy_batch.meta_info["logprobs"] = 20
        entropy_batch.meta_info["max_tokens"] = 256
        entropy_batch.meta_info["n"] = 1

        try:
            entropy_output = self.actor_rollout_wg.generate_sequences(entropy_batch)
        except Exception as e:
            logger.warning(f"[Evolution] entropy generation failed: {e}")
            entropy_output = None

        if entropy_output is not None:
            all_h = self._compute_entropy_from_logprobs(entropy_output, n_children)
        else:
            all_h = [None] * n_children

        # ================================================================
        # Phase 4: Scoring + Grid insertion
        # ================================================================
        attempted = 0
        inserted = 0
        for (child, inst, op), flags, h_bar in zip(children, all_flags, all_h):
            attempted += 1
            p_hat = sum(flags) / len(flags) if flags else 0.0

            if h_bar is None or not h_prefilter(h_bar, self.h_threshold):
                logger.debug(f"[Evolution] H={h_bar} below threshold, skip")
                continue

            if not p_hat_filter(p_hat):
                logger.debug(f"[Evolution] p_hat={p_hat:.2f} extreme, skip")
                continue

            rq_result = compute_rq_full(flags, h_bar)
            child.p_hat = p_hat
            child.h_score = h_bar
            child.rq_score = rq_result.rq_score
            child.fitness = rq_result.rq_score

            was_inserted = self.map_elites.try_insert(
                program=child, h_value=h_bar,
                problem_text=inst.problem, rq_score=rq_result.rq_score,
            )
            if was_inserted:
                inserted += 1
                logger.info(
                    f"[Evolution] Inserted ({op}): p={p_hat:.3f} "
                    f"H={h_bar:.3f} R_Q={rq_result.rq_score:.4f}"
                )

        return attempted, inserted

    # ------------------------------------------------------------------
    # Exact entropy: actor_rollout_wg.compute_log_prob(calculate_entropy=True)
    # ------------------------------------------------------------------

    def _compute_entropy_from_logprobs(
        self, entropy_output: DataProto, n_children: int, top_k: int = 20,
    ) -> list[float | None]:
        """
        generate_sequences의 logprobs 출력에서 Shannon entropy 근사 계산.
        Feasibility test의 VLLMRunner.batch_entropy()와 동일한 방식.

        H_t = -Σ_{v∈top-K} p(v) log p(v) + rest mass 보정
        """
        import math

        vocab_size = self.tokenizer.vocab_size or 150000
        results = []

        resp_ids = entropy_output.batch.get("responses")
        if resp_ids is None:
            return [None] * n_children

        # logprobs가 output에 포함되어 있는지 확인
        logprobs_data = entropy_output.batch.get("logprobs")

        if logprobs_data is None:
            # logprobs가 없으면 old_log_probs에서 근사
            old_lp = entropy_output.batch.get("old_log_probs")
            if old_lp is not None:
                for ci in range(n_children):
                    lps = old_lp[ci]
                    mask = entropy_output.batch.get("response_mask")
                    if mask is not None:
                        valid_lps = lps[mask[ci].bool()]
                    else:
                        valid_lps = lps[lps != 0]
                    if valid_lps.numel() > 0:
                        h = -valid_lps.float().mean().item()
                        results.append(max(0.0, h))
                    else:
                        results.append(None)
            else:
                # 둘 다 없으면 binary entropy proxy 사용
                logger.debug("[Evolution] No logprobs available, using binary entropy proxy")
                results = [None] * n_children
        else:
            # top-K logprobs에서 Shannon entropy 계산
            for ci in range(n_children):
                try:
                    step_logprobs = logprobs_data[ci]  # 토큰별 top-K logprobs
                    token_hs = []
                    for step_dict in step_logprobs:
                        if step_dict is None:
                            continue
                        log_probs = [lp.logprob for lp in step_dict.values()]
                        probs = [math.exp(lp) for lp in log_probs]
                        h_top = -sum(p * math.log(p) for p in probs if p > 0)
                        p_rest = max(0.0, 1.0 - sum(probs))
                        if p_rest > 1e-8 and vocab_size > top_k:
                            h_top += -p_rest * math.log(p_rest / (vocab_size - top_k))
                        token_hs.append(h_top)
                    if token_hs:
                        results.append(sum(token_hs) / len(token_hs))
                    else:
                        results.append(None)
                except Exception:
                    results.append(None)

        return results

    # ------------------------------------------------------------------
    # Dataset refresh from MAP-Elites champions
    # ------------------------------------------------------------------

    def _refresh_dataset(self):
        """MAP-Elites 챔피언 → dynamic_dataset 교체 → dataloader 재구성."""
        champions = self.map_elites.get_all_champions()
        old_size = len(self.dynamic_dataset)
        new_problems: list[dict] = []

        for champ in champions:
            for seed in range(self.instances_per_program):
                inst = champ.execute(seed=seed)
                if inst:
                    new_problems.append({
                        "problem": inst.problem,
                        "answer": inst.answer,
                        "program_id": champ.program_id,
                        "rq_score": champ.rq_score,
                    })

        if new_problems:
            self.dynamic_dataset.update(new_problems)
            self._rebuild_dataloader()
            logger.info(
                f"[Evolution] Dataset refreshed: {old_size} → {len(new_problems)} problems "
                f"from {len(champions)} champions, dataloader rebuilt"
            )

    def _rebuild_dataloader(self):
        """dynamic_dataset 변경 후 train_dataloader를 재구성.

        현재 epoch의 iterator는 이미 생성되어 있으므로 즉시 반영되지 않지만,
        num_workers=0 + __getitem__의 thread-safe update 덕분에
        데이터 내용 자체는 즉시 반영된다.
        다음 epoch부터는 새 dataloader의 sampler 범위가 적용된다.
        """
        self.train_dataloader = StatefulDataLoader(
            dataset=self.dynamic_dataset,
            batch_size=self.config.data.rollout_batch_size,
            num_workers=0,
            drop_last=True,
            collate_fn=verl_collate_fn,
        )
