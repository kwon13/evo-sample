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
import uuid
import random
import logging
import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .map_elites import MAPElitesGrid
from .program import ProblemProgram
from .rq_score import compute_rq_full, h_prefilter
from .verl_dataset import MapElitesDynamicDataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)

IN_DEPTH_PROMPT = (
    "You are an expert mathematician and Python programmer.\n\n"
    "Below is a Python function that generates natural-language math word problems "
    "using inverse construction (answer chosen first, problem built from it).\n\n"
    "```python\n{source_code}\n```\n\n"
    "Modify this function to generate HARDER problems. You may:\n"
    "- Add more reasoning steps or constraints\n"
    "- Combine multiple math concepts\n"
    "- Require multi-step logic\n\n"
    "RULES:\n"
    "1. Function MUST be named `generate` and take a single `seed` argument\n"
    "2. MUST return (problem_text: str, answer: str)\n"
    "3. answer MUST be constructed FIRST, then problem built from it\n"
    "4. Use only standard library + math module\n"
    "5. Self-contained, no external dependencies\n\n"
    "Return ONLY the Python code."
)

IN_BREADTH_PROMPT = (
    "You are an expert mathematician and Python programmer.\n\n"
    "Below is a Python function that generates math word problems:\n\n"
    "```python\n{source_code}\n```\n\n"
    "Create a COMPLETELY DIFFERENT type of math word problem generator covering "
    "a different topic.\n\n"
    "RULES:\n"
    "1. Function MUST be named `generate` and take a single `seed` argument\n"
    "2. MUST return (problem_text: str, answer: str)\n"
    "3. answer MUST be constructed FIRST, then problem built from it\n"
    "4. Use only standard library + math module\n"
    "5. Self-contained\n\n"
    "Return ONLY the Python code."
)

_BOXED_RE = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def _normalize(s: str) -> str:
    s = s.strip().lower()
    for o, c in [("{", "}"), ("[", "]"), ("(", ")")]:
        if s.startswith(o) and s.endswith(c):
            s = s[1:-1].strip()
    return " ".join(s.rstrip(".").split())


def _answers_match(pred: str, gt: str) -> bool:
    if _normalize(pred) == _normalize(gt):
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
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
        candidates_per_evo: int = 8,
        num_rollouts: int = 16,
        instances_per_program: int = 3,
        in_depth_ratio: float = 0.7,
        h_threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.map_elites = map_elites
        self.dynamic_dataset = dynamic_dataset
        self.evolution_freq = evolution_freq
        self.candidates_per_evo = candidates_per_evo
        self.num_rollouts = num_rollouts
        self.instances_per_program = instances_per_program
        self.in_depth_ratio = in_depth_ratio
        self.h_threshold = h_threshold
        self._update_actor_call_count = 0

    # ------------------------------------------------------------------
    # Hook: called after every GRPO mini-batch update
    # ------------------------------------------------------------------

    def _update_actor(self, batch: DataProto) -> DataProto:
        """GRPO update → evolution step every evolution_freq calls."""
        actor_output = super()._update_actor(batch)

        self._update_actor_call_count += 1
        if self._update_actor_call_count % self.evolution_freq == 0:
            evo_metrics = self._evolution_step()
            # Attach evolution metrics to actor_output so they appear in logs
            existing = actor_output.meta_info.get("metrics", {})
            existing.update({f"evo/{k}": v for k, v in evo_metrics.items()})
            actor_output.meta_info["metrics"] = existing

        return actor_output

    # ------------------------------------------------------------------
    # Evolution step (driver-side, CPU)
    # ------------------------------------------------------------------

    def _evolution_step(self) -> dict:
        """
        Evolution 1회:
          1. MAP-Elites에서 parent 샘플링
          2. async_rollout_manager로 mutation 생성 (questioner role)
          3. 프로그램 실행 → (problem, answer)
          4. async_rollout_manager로 G rollout (solver role) → p_hat
          5. actor_rollout_wg.compute_log_prob(calculate_entropy=True) → H̄
          6. R_Q = p(1-p)*H̄ → MAP-Elites 갱신
          7. dynamic_dataset 갱신
        """
        logger.info(
            f"[Evolution] step at actor_update #{self._update_actor_call_count}"
        )

        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        temperature = self.config.actor_rollout_ref.rollout.temperature

        inserted = 0
        attempted = 0

        for _ in range(self.candidates_per_evo):
            parent = self.map_elites.sample_parent()
            if parent is None:
                continue

            # ---- Step 1: Mutation generation ----
            mtype = "in_depth" if random.random() < self.in_depth_ratio else "in_breadth"
            tmpl = IN_DEPTH_PROMPT if mtype == "in_depth" else IN_BREADTH_PROMPT
            mut_text = tmpl.format(source_code=parent.source_code)

            mut_batch = _make_gen_batch(
                raw_prompts=[[{"role": "user", "content": mut_text}]],
                answers=[""],
                temperature=temperature,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                global_steps=self.global_steps,
            )

            try:
                mut_output = self.async_rollout_manager.generate_sequences(mut_batch)
            except Exception as e:
                logger.warning(f"[Evolution] mutation generate failed: {e}")
                continue

            # Decode generated code
            resp_ids = mut_output.batch.get("responses")
            if resp_ids is None:
                continue
            code_text = self.tokenizer.decode(
                resp_ids[0].tolist(), skip_special_tokens=True
            )
            source_code = _extract_code(code_text)
            if not source_code:
                continue

            # ---- Step 2: Execute program ----
            child = ProblemProgram(
                source_code=source_code,
                parent_id=parent.program_id,
                generation=parent.generation + 1,
                metadata={"mutation_type": mtype},
            )
            inst = child.execute(seed=42, timeout=5.0)
            if inst is None:
                continue

            attempted += 1
            problem_text = inst.problem
            answer_text = inst.answer

            # ---- Step 3: G rollouts for p_hat ----
            solver_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem_text},
            ]
            rollout_batch = _make_gen_batch(
                raw_prompts=[solver_prompt],
                answers=[answer_text],
                temperature=temperature,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                global_steps=self.global_steps,
                n_repeat=self.num_rollouts,
            )

            try:
                rollout_output = self.async_rollout_manager.generate_sequences(
                    rollout_batch
                )
            except Exception as e:
                logger.warning(f"[Evolution] rollout generate failed: {e}")
                continue

            resp_ids = rollout_output.batch.get("responses")
            if resp_ids is None:
                continue

            correct_flags = []
            for i in range(resp_ids.shape[0]):
                decoded = self.tokenizer.decode(
                    resp_ids[i].tolist(), skip_special_tokens=True
                )
                pred = _extract_boxed(decoded)
                correct_flags.append(
                    _answers_match(pred, answer_text) if pred else False
                )
            p_hat = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0

            # ---- Step 4: Exact entropy via FSDP actor ----
            # rollout_output에는 이미 (prompt + response) 토큰이 있다.
            # compute_log_prob(calculate_entropy=True)가 전체 vocab logits에서
            # H = -Σ p_v log p_v 를 계산한다 (verl_F.entropy_from_logits).
            h_bar = self._compute_exact_entropy(rollout_output)

            if h_bar is None or not h_prefilter(h_bar, self.h_threshold):
                logger.debug(f"[Evolution] H={h_bar} below threshold, skip")
                continue

            # ---- Step 5: R_Q & MAP-Elites update ----
            rq_result = compute_rq_full(correct_flags, h_bar)

            child.p_hat = p_hat
            child.h_score = h_bar
            child.rq_score = rq_result.rq_score
            child.fitness = rq_result.rq_score

            was_inserted = self.map_elites.try_insert(
                program=child,
                h_value=h_bar,
                problem_text=problem_text,
                rq_score=rq_result.rq_score,
            )
            if was_inserted:
                inserted += 1
                logger.info(
                    f"[Evolution] Inserted: p_hat={p_hat:.3f} "
                    f"H={h_bar:.3f} R_Q={rq_result.rq_score:.4f}"
                )

        # ---- Step 6: Refresh training dataset ----
        self._refresh_dataset()

        stats = self.map_elites.stats()
        return {
            "attempted": attempted,
            "inserted": inserted,
            "grid_coverage": stats["coverage"],
            "grid_mean_rq": stats["mean_rq"],
            "grid_max_rq": stats["max_rq"],
            "grid_champions": stats["num_champions"],
        }

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
