"""
RQ-Evolve Trainer: veRL RayPPOTrainer + MAP-Elites evolution.

설계 원칙:
  - 모델은 단 한 번 GPU에 로드 (vLLM rollout + FSDP actor 공유)
  - RayPPOTrainer.fit()를 그대로 사용하고 _update_actor()만 override
  - _update_actor()가 GRPO step을 수행한 뒤 evolution_freq마다 _evolution_step() 호출
  - evolution_step 내부:
      1. generate_sequences (mutation prompt)    → 변이 Python 코드
      2. generate_sequences (solver, n=G)        → p_hat 추정
      3. actor_rollout_wg.compute_log_prob(      → 정확한 Shannon entropy
             calculate_entropy=True)               (전체 vocab logits, FSDP actor)
      4. R_Q 계산 → MAP-Elites 갱신
      5. dynamic_dataset 갱신

vLLM logprobs=1 근사:  H ≈ -log p_top        (이전 방식)
FSDP actor forward:    H = -Σ_v p_v log p_v   (이 방식, 정확)
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

from verl.protocol import DataProto
from verl.trainer.ray_trainer import RayPPOTrainer

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
    raw_prompts: list[list[dict]],
    answers: list[str],
    temperature: float,
    eos_token_id: int,
    pad_token_id: int,
    global_steps: int,
    n_repeat: int = 1,
) -> DataProto:
    """
    SingleTurnAgentLoop이 기대하는 DataProto 포맷 생성.
    raw_prompt: messages list (apply_chat_template은 AgentLoop 내부에서 수행)
    """
    B = len(raw_prompts)
    batch = DataProto(
        batch={"dummy_tensor": torch.zeros(B, dtype=torch.uint8)},
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "data_source": np.array(["rq_evolved"] * B, dtype=object),
            "reward_model": np.array(
                [{"ground_truth": a} for a in answers], dtype=object
            ),
            "extra_info": np.array([{}] * B, dtype=object),
            "uid": np.array(
                [str(uuid.uuid4()) for _ in range(B)], dtype=object
            ),
        },
    )
    batch.meta_info = {
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "temperature": temperature,
        "do_sample": True,
        "global_steps": global_steps,
    }
    if n_repeat > 1:
        batch = batch.repeat(repeat_times=n_repeat, interleave=True)
    return batch


# ---------------------------------------------------------------------------
# RQEvolveTrainer
# ---------------------------------------------------------------------------

class RQEvolveTrainer(RayPPOTrainer):
    """
    RayPPOTrainer + MAP-Elites evolution.

    _update_actor()를 override해서 evolution_freq마다 _evolution_step() 삽입.
    fit() 자체는 RayPPOTrainer 그대로 사용 (재구현 없음).
    """

    def __init__(
        self,
        *args,
        map_elites: MAPElitesGrid,
        dynamic_dataset: MapElitesDynamicDataset,
        evolution_freq: int = 50,
        evolution_pct: float | None = None,
        candidates_per_evo: int = 8,
        num_rollouts: int = 16,
        instances_per_program: int = 3,
        in_depth_ratio: float = 0.5,
        crossover_ratio: float = 0.2,
        h_threshold: float = 0.1,
        target_hard_champions: int = 6,
        max_evo_attempts: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.map_elites = map_elites
        self.dynamic_dataset = dynamic_dataset
        self.evolution_freq = evolution_freq
        self.evolution_pct = evolution_pct
        self.candidates_per_evo = candidates_per_evo
        self.num_rollouts = num_rollouts
        self.instances_per_program = instances_per_program
        self.in_depth_ratio = in_depth_ratio
        self.crossover_ratio = crossover_ratio
        self.h_threshold = h_threshold
        self.target_hard_champions = target_hard_champions
        self.max_evo_attempts = max_evo_attempts
        self._update_actor_call_count = 0
        self._computed_evolution_freq = None   # pct 기반 계산 결과 (lazy)
        self._last_evo_metrics = None

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

    def _update_actor(self, batch: DataProto) -> DataProto:
        """
        Evolution-first ordering:
          1. evolution_freq마다 _evolution_step() 실행 (dataset 교체)
          2. 그 다음 GRPO/REINFORCE++ update (새 문제로 학습)

        count=0에서 첫 evolution 실행 → 시드에서 바로 진화.
        """
        freq = self._get_evolution_freq()

        # Evolution FIRST
        if self._update_actor_call_count % freq == 0:
            evo_metrics = self._evolution_step()
            self._last_evo_metrics = evo_metrics

        # Then RL update (GRPO or REINFORCE++)
        actor_output = super()._update_actor(batch)
        self._update_actor_call_count += 1

        # Attach evo metrics to the step where evolution ran
        if self._last_evo_metrics:
            existing = actor_output.meta_info.get("metrics", {})
            existing.update({f"evo/{k}": v for k, v in self._last_evo_metrics.items()})
            actor_output.meta_info["metrics"] = existing
            self._last_evo_metrics = None

        return actor_output

    # ------------------------------------------------------------------
    # Evolution step (driver-side, CPU)
    # ------------------------------------------------------------------

    def _evolution_step(self) -> dict:
        """
        Batch-filling evolution (Bae et al. 2025, Appendix B 스타일).

        target_champions 이상 확보되거나 max_evo_attempts에 도달할 때까지
        _evolution_round()를 반복. 최종 1회 _refresh_dataset().
        """
        logger.info(
            f"[Evolution] step at actor_update #{self._update_actor_call_count} "
            f"(target_hard={self.target_hard_champions}, max_attempts={self.max_evo_attempts})"
        )

        total_attempted = 0
        total_inserted = 0
        round_num = 0

        while True:
            # 최소 1라운드 반드시 실행, 이후 target/max 체크
            if round_num > 0:
                hard = self.map_elites.count_hard_champions(min_h_bin=2)
                if hard >= self.target_hard_champions:
                    logger.info(f"[Evolution] H2+ target reached: {hard} >= {self.target_hard_champions}")
                    break
                if total_attempted >= self.max_evo_attempts:
                    logger.info(f"[Evolution] Max attempts reached: {total_attempted}")
                    break

            batch_size = min(
                self.candidates_per_evo,
                self.max_evo_attempts - total_attempted,
            )
            round_num += 1
            hard = self.map_elites.count_hard_champions(min_h_bin=2)
            logger.info(
                f"[Evolution] Round {round_num}: {batch_size} candidates "
                f"(H2+ champions={hard}/{self.target_hard_champions})"
            )

            attempted, inserted = self._evolution_round(batch_size)
            total_attempted += attempted
            total_inserted += inserted

        self._refresh_dataset()

        stats = self.map_elites.stats()
        return {
            "attempted": total_attempted,
            "inserted": total_inserted,
            "rounds": round_num,
            "grid_coverage": stats["coverage"],
            "grid_mean_rq": stats["mean_rq"],
            "grid_max_rq": stats["max_rq"],
            "grid_champions": stats["num_champions"],
            "hard_champions": stats["hard_champions"],
        }

    def _evolution_round(self, batch_size: int) -> tuple[int, int]:
        """
        단일 라운드: batch_size개 candidate mutation → rollout → entropy → grid insert.
        _refresh_dataset()는 호출하지 않음 (caller가 최종 1회 호출).

        Returns: (attempted, inserted)
        """
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        temperature = self.config.actor_rollout_ref.rollout.temperature

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
            raw_prompts=[
                [{"role": "user", "content": pt}] for pt, _, _, _ in mutation_prompts
            ],
            answers=[""] * len(mutation_prompts),
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            global_steps=self.global_steps,
        )

        try:
            mut_output = self.async_rollout_manager.generate_sequences(mut_batch)
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
                    root_seed_id=parent.root_seed_id,  # parent_a의 계열 계승
                    metadata={"op": "crossover"},
                )
            else:
                child = ProblemProgram(
                    source_code=source_code,
                    parent_id=parent.program_id,
                    generation=parent.generation + 1,
                    root_seed_id=parent.root_seed_id,  # 부모의 시드 계열 계승
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
        solver_prompts = [
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": inst.problem}]
            for _, inst, _ in children
        ]
        rollout_batch = _make_gen_batch(
            raw_prompts=solver_prompts,
            answers=[inst.answer for _, inst, _ in children],
            temperature=temperature,
            eos_token_id=eos_id, pad_token_id=pad_id,
            global_steps=self.global_steps,
            n_repeat=self.num_rollouts,
        )

        try:
            rollout_output = self.async_rollout_manager.generate_sequences(rollout_batch)
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
        # Phase 3: Batch entropy (첫 rollout 응답 사용)
        # ================================================================
        entropy_indices = [ci * self.num_rollouts for ci in range(n_children)]
        entropy_slice = rollout_output[entropy_indices]
        all_h = []
        for ci in range(n_children):
            single = entropy_slice[ci:ci+1]
            h = self._compute_exact_entropy(single)
            all_h.append(h)

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

    def _compute_exact_entropy(self, rollout_output: DataProto) -> float | None:
        """
        rollout_output: generate_sequences가 반환한 DataProto
          (input_ids, responses, attention_mask, position_ids 포함)

        actor_rollout_wg.compute_log_prob(calculate_entropy=True)를 호출.
        이 메서드는 FSDP actor의 전체 vocab logits에서
        H_t = -Σ_v p_v log p_v 를 계산해 response 토큰별로 반환한다.

        반환: 응답 토큰 평균 entropy (scalar float)
        """
        try:
            # 첫 번째 샘플만 사용해 compute 비용 절감
            single = rollout_output[0:1]

            # compute_log_prob에 필요한 meta_info
            rollout_cfg = self.config.actor_rollout_ref.rollout
            single.meta_info = {
                "micro_batch_size": 1,
                "use_dynamic_bsz": False,
                "temperature": rollout_cfg.temperature,
                "pad_token_id": self.tokenizer.pad_token_id or 0,
            }

            entropy_out = self.actor_rollout_wg.compute_log_prob(
                single, calculate_entropy=True
            )

            # compute_log_prob은 dict 또는 DataProto를 반환할 수 있다
            if isinstance(entropy_out, dict):
                entropys = entropy_out.get("entropys")
            else:
                entropys = (
                    entropy_out.batch.get("entropys")
                    if hasattr(entropy_out, "batch")
                    else None
                )

            if entropys is None:
                logger.debug("[Evolution] entropys not in compute_log_prob output")
                return None

            # entropys shape: (1, response_len) 또는 (response_len,)
            if entropys.dim() == 2:
                entropys = entropys[0]  # (response_len,)

            # response_mask로 padding 제외
            batch_data = single.batch if hasattr(single, "batch") else {}
            if "response_mask" in batch_data:
                mask = batch_data["response_mask"][0].bool()
                valid = entropys[: mask.shape[0]][mask]
                h = valid.mean().item() if valid.numel() > 0 else entropys.mean().item()
            else:
                h = entropys.mean().item()

            return max(0.0, h)

        except Exception as e:
            logger.warning(f"[Evolution] _compute_exact_entropy failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Dataset refresh from MAP-Elites champions
    # ------------------------------------------------------------------

    def _refresh_dataset(self):
        """MAP-Elites 챔피언 → dynamic_dataset 교체."""
        champions = self.map_elites.get_all_champions()
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
            logger.info(
                f"[Evolution] Dataset refreshed: {len(new_problems)} problems "
                f"from {len(champions)} champions"
            )
