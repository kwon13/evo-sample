"""
Evolution Feasibility Test (veRL-free)

veRL 없이 evolution 로직 전체의 동작 가능성을 검증한다.

모드:
  1. 순수 Mock    : LLM/entropy/rollout 모두 시뮬레이션
  2. OpenAI API  : --model gpt-4o-mini 등으로 mutation만 실제 LLM
  3. vLLM 실제   : --vllm_model Qwen/Qwen3-8B-Base
                   mutation / rollout / entropy 모두 실제 모델로 동작

사용법:
  # 순수 mock:
  python scripts/test_feasibility.py

  # vLLM 실제 시나리오 (권장):
  python scripts/test_feasibility.py \
      --vllm_model Qwen/Qwen3-8B-Base \
      --n_evo 10 --candidates 8

  # OpenAI-compatible API (mutation만):
  python scripts/test_feasibility.py \
      --model gpt-4o-mini \
      --base_url https://api.openai.com/v1 \
      --api_key sk-...

  # 빠른 smoke test:
  python scripts/test_feasibility.py --n_evo 3 --candidates 4
"""

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import random
import re
import sys
import time
import textwrap
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rq_questioner.program import ProblemProgram, ProblemInstance
from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.rq_score import compute_rq_full, h_prefilter, p_hat_filter
from prompts import (
    MUTATE_DEPTH, MUTATE_BREADTH, MUTATE_CROSSOVER,
    SINGLE_ANSWER_RULE, SCORE_FEEDBACK,
    score_diagnosis, build_score_feedback,
    build_few_shot_examples, build_execution_feedback,
    SOLVER_COMPLETION_PROMPT,
)
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Answer extraction helpers (rollout 정답 비교용)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Self-verification + Few-shot + Execution feedback
# ---------------------------------------------------------------------------

def _verify_program(program: ProblemProgram, n_seeds: int = 5) -> Optional[ProblemInstance]:
    """Multi-seed 실행 + SymPy 답 유효성 검증."""
    from sympy import sympify
    instances = []
    for s in range(n_seeds):
        inst = program.execute(seed=s, timeout=5.0)
        if inst is None:
            return None
        try:
            sympify(inst.answer.strip().replace("^", "**"))
        except Exception:
            try:
                float(inst.answer)
            except (ValueError, TypeError):
                return None
        instances.append(inst)
    return instances[0] if instances else None






_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def _extract_boxed(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def _sympy_equal(a: str, b: str, tol: float = 1e-4) -> Optional[bool]:
    """
    SymPy로 두 수학 표현식이 동치인지 판별.

    지원 범위:
      - 분수: 31/45, 3/4
      - 제곱근: sqrt(2), 2*sqrt(3)
      - 상수: pi, e
      - 거듭제곱: 2**10, 2^10
      - 사칙연산: 2*3+1, (1+2)/3
      - 소수: 0.6889, 3.14
      - LaTeX 스타일: \\frac{3}{4}, \\sqrt{2}

    Returns:
      True/False if comparison succeeded, None if parsing failed.
    """
    from sympy import sympify, N, simplify
    from sympy.parsing.latex import parse_latex

    def _parse(s: str):
        s = s.strip()
        # ^ → ** (거듭제곱 표기 통일)
        s_py = s.replace("^", "**")
        # LaTeX 시도 (\frac, \sqrt 등)
        if "\\" in s:
            try:
                return parse_latex(s)
            except Exception:
                pass
        # SymPy sympify
        try:
            return sympify(s_py)
        except Exception:
            pass
        return None

    expr_a = _parse(a)
    expr_b = _parse(b)
    if expr_a is None or expr_b is None:
        return None

    # 1. 기호적 동치 (exact)
    try:
        if simplify(expr_a - expr_b) == 0:
            return True
    except Exception:
        pass

    # 2. 수치적 비교 (float 근사)
    try:
        val_a = float(N(expr_a))
        val_b = float(N(expr_b))
        return abs(val_a - val_b) < tol
    except Exception:
        pass

    return None


def _answers_match(pred: str, gt: str, tol: float = 1e-2) -> bool:
    # 1. 문자열 정확 일치
    if pred.strip().lower() == gt.strip().lower():
        return True
    # 2. SymPy 기반 수학적 동치 판별
    result = _sympy_equal(pred, gt, tol=tol)
    if result is not None:
        return result
    return False


# ---------------------------------------------------------------------------
# vLLM 실제 백엔드 (mutation / rollout / entropy)
# ---------------------------------------------------------------------------

_SOLVE_PROMPT = SOLVER_COMPLETION_PROMPT





class VLLMRunner:
    """
    vLLM 기반 실제 mutation / rollout / entropy 측정.

    entropy 측정 방식:
      vLLM generate()의 logprobs=1 옵션으로 각 토큰의 top-1 log prob을 얻어,
      H ≈ -1/T × Σ log p(token_t) 로 근사.
      (full vocab Shannon entropy가 아닌 cross-entropy proxy이지만
       실제 pipeline.py의 _measure_h_batch()와 동일한 방식.)
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        from vllm import LLM
        print(f"[vLLM] Loading {model_name} ...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.temperature = temperature
        self.max_tokens = max_tokens
        print(f"[vLLM] Model loaded.")

    # ---- entropy --------------------------------------------------------

    def entropy(self, inst: "ProblemInstance", top_k: int = 20) -> Optional[float]:
        """
        Shannon entropy 근사: top-K logprobs 기반.

        각 토큰 위치에서 top-K 토큰의 log probability를 받아
        H_t = -Σ_{v∈top-K} p(v) log p(v) 로 계산.

        top-K 밖 확률 mass를 균등 분배하여 보정:
          p_rest = 1 - Σ p(top-K)
          H_rest = p_rest * log(vocab_size - K)  (upper bound)

        이 방식은 veRL의 entropy_from_logits (full vocab)보다
        근사적이지만, -logprob(top-1) proxy보다 훨씬 정확함.
        """
        import math
        from vllm import SamplingParams

        prompt = _SOLVE_PROMPT.format(problem=inst.problem)
        params = SamplingParams(
            temperature=1.0,   # entropy 측정은 temperature=1.0에서 수행
            max_tokens=min(self.max_tokens, 256),  # entropy는 앞부분만으로 충분
            logprobs=top_k,
            n=1,
        )
        out = self.llm.generate([prompt], params)[0].outputs[0]
        if not out.logprobs:
            return None

        vocab_size = self.tokenizer.vocab_size or 150000
        token_entropies = []

        for step_dict in out.logprobs:
            # step_dict: {token_id: LogProb, ...} — top-K개
            log_probs = [lp.logprob for lp in step_dict.values()]
            probs = [math.exp(lp) for lp in log_probs]

            # top-K 토큰의 entropy
            h_top = -sum(p * math.log(p) for p in probs if p > 0)

            # top-K 밖의 잔여 확률 mass 보정
            p_rest = max(0.0, 1.0 - sum(probs))
            if p_rest > 1e-8 and vocab_size > top_k:
                # 잔여 mass를 (vocab_size - K)개에 균등 분배 (upper bound)
                h_rest = -p_rest * math.log(p_rest / (vocab_size - top_k))
                h_top += h_rest

            token_entropies.append(h_top)

        return sum(token_entropies) / len(token_entropies) if token_entropies else None

    # ---- rollout --------------------------------------------------------

    def rollout(self, inst: "ProblemInstance", n_rollouts: int) -> tuple[list[bool], list[dict]]:
        """
        G번 생성 → boxed 정답 추출 → 정오 판별.
        Returns: (flags, rollout_logs)
        """
        from vllm import SamplingParams
        prompt = _SOLVE_PROMPT.format(problem=inst.problem)
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n_rollouts,
        )
        outputs = self.llm.generate([prompt], params)[0].outputs
        flags = []
        logs = []
        for i, comp in enumerate(outputs):
            pred = _extract_boxed(comp.text)
            correct = _answers_match(pred, inst.answer) if pred else False
            flags.append(correct)
            logs.append({
                "rollout_idx": i,
                "response": comp.text,
                "extracted": pred,
                "ground_truth": inst.answer,
                "correct": correct,
            })
        return flags, logs

    # ---- mutation -------------------------------------------------------

    def mutate(
        self, parent: "ProblemProgram", in_depth: bool = True,
        grid: Optional["MAPElitesGrid"] = None,
    ) -> Optional[str]:
        """
        Score-aware mutation + few-shot + execution feedback.
        """
        from vllm import SamplingParams
        tmpl = MUTATE_DEPTH if in_depth else MUTATE_BREADTH

        p_hat = getattr(parent, "p_hat", 0.5)
        h_score = getattr(parent, "h_score", 1.0)
        rq_score = getattr(parent, "rq_score", 0.0)
        diag, action = score_diagnosis(p_hat, h_score)

        score_feedback = SCORE_FEEDBACK.format(
            p_hat=p_hat, h_score=h_score, rq_score=rq_score,
            diagnosis=diag, action=action,
        )
        few_shot = build_few_shot_examples(grid) if grid else ""
        exec_fb = build_execution_feedback(parent)
        prompt = tmpl.format(
            code=parent.source_code, score_feedback=score_feedback,
            few_shot=few_shot, exec_feedback=exec_fb,
        )
        params = SamplingParams(
            temperature=0.1,   # 코드 생성: 낮은 temperature → syntax 오류 감소
            max_tokens=4096,
            n=1,
        )
        text = self.llm.generate([prompt], params)[0].outputs[0].text

        # 프롬프트에 "import random\n" 까지 포함됐으므로 앞에 붙여줌
        full_code = "import random\n" + text

        # ``` 이후 내용 제거 (닫는 코드블록)
        cut = full_code.find("```")
        if cut != -1:
            full_code = full_code[:cut].strip()

        return full_code if "def generate" in full_code else None

    # ---- crossover ------------------------------------------------------

    def crossover(
        self, parent_a: "ProblemProgram", parent_b: "ProblemProgram",
        grid: Optional["MAPElitesGrid"] = None,
    ) -> Optional[str]:
        """
        두 부모 프로그램의 수학 개념을 결합하여 새 프로그램 생성.
        MAP-Elites 원본(Mouret & Clune, 2015)의 crossover 연산자.
        """
        from vllm import SamplingParams

        few_shot = build_few_shot_examples(grid) if grid else ""
        prompt = MUTATE_CROSSOVER.format(
            code_a=parent_a.source_code,
            code_b=parent_b.source_code,
            p_hat_a=getattr(parent_a, "p_hat", 0.5),
            h_a=getattr(parent_a, "h_score", 1.0),
            p_hat_b=getattr(parent_b, "p_hat", 0.5),
            h_b=getattr(parent_b, "h_score", 1.0),
            few_shot=few_shot,
        )
        params = SamplingParams(
            temperature=0.3,
            max_tokens=4096,
            n=1,
        )
        text = self.llm.generate([prompt], params)[0].outputs[0].text
        full_code = "import random\n" + text
        cut = full_code.find("```")
        if cut != -1:
            full_code = full_code[:cut].strip()
        return full_code if "def generate" in full_code else None

    # ---- batch methods (논문 Appendix B 스타일) --------------------------------

    def _extract_code(self, text: str) -> Optional[str]:
        """Mutation/crossover 출력에서 코드 추출."""
        full_code = "import random\n" + text
        cut = full_code.find("```")
        if cut != -1:
            full_code = full_code[:cut].strip()
        return full_code if "def generate" in full_code else None

    def batch_mutate(
        self, tasks: list[dict], grid: Optional["MAPElitesGrid"] = None,
    ) -> list[Optional[str]]:
        """
        Batch mutation: 여러 mutation/crossover 프롬프트를 한 번의 generate()로 처리.
        Few-shot top-K 예시 + execution feedback 포함.

        tasks: list of {"op": "in_depth"|"in_breadth"|"crossover", "parent": ..., "parent_b": ...}
        Returns: list of Optional[str] (source code or None)
        """
        from vllm import SamplingParams

        few_shot = build_few_shot_examples(grid) if grid else ""

        prompts = []
        for t in tasks:
            if t["op"] == "crossover":
                pa, pb = t["parent"], t["parent_b"]
                prompts.append(MUTATE_CROSSOVER.format(
                    code_a=pa.source_code, code_b=pb.source_code,
                    p_hat_a=getattr(pa, "p_hat", 0.5), h_a=getattr(pa, "h_score", 1.0),
                    p_hat_b=getattr(pb, "p_hat", 0.5), h_b=getattr(pb, "h_score", 1.0),
                    few_shot=few_shot,
                ))
            else:
                parent = t["parent"]
                tmpl = MUTATE_DEPTH if t["op"] == "in_depth" else MUTATE_BREADTH
                p_hat = getattr(parent, "p_hat", 0.5)
                h_score = getattr(parent, "h_score", 1.0)
                rq_score = getattr(parent, "rq_score", 0.0)
                diag, action = score_diagnosis(p_hat, h_score)
                feedback = SCORE_FEEDBACK.format(
                    p_hat=p_hat, h_score=h_score, rq_score=rq_score,
                    diagnosis=diag, action=action,
                )
                exec_fb = build_execution_feedback(parent)
                prompts.append(tmpl.format(
                    code=parent.source_code, score_feedback=feedback,
                    few_shot=few_shot, exec_feedback=exec_fb,
                ))

        if not prompts:
            return []

        params = SamplingParams(temperature=0.3, max_tokens=4096, n=1)
        outputs = self.llm.generate(prompts, params)
        return [self._extract_code(out.outputs[0].text) for out in outputs]

    def batch_entropy(
        self, instances: list["ProblemInstance"], top_k: int = 20,
    ) -> list[Optional[float]]:
        """Batch entropy 측정: 여러 문제를 한 번의 generate()로 처리."""
        import math as _math
        from vllm import SamplingParams

        prompts = [_SOLVE_PROMPT.format(problem=inst.problem) for inst in instances]
        if not prompts:
            return []

        params = SamplingParams(
            temperature=1.0,
            max_tokens=min(self.max_tokens, 256),
            logprobs=top_k, n=1,
        )
        outputs = self.llm.generate(prompts, params)
        vocab_size = self.tokenizer.vocab_size or 150000

        results = []
        for out in outputs:
            lp_list = out.outputs[0].logprobs
            if not lp_list:
                results.append(None)
                continue
            token_hs = []
            for step_dict in lp_list:
                log_probs = [lp.logprob for lp in step_dict.values()]
                probs = [_math.exp(lp) for lp in log_probs]
                h_top = -sum(p * _math.log(p) for p in probs if p > 0)
                p_rest = max(0.0, 1.0 - sum(probs))
                if p_rest > 1e-8 and vocab_size > top_k:
                    h_top += -p_rest * _math.log(p_rest / (vocab_size - top_k))
                token_hs.append(h_top)
            results.append(sum(token_hs) / len(token_hs) if token_hs else None)
        return results

    def batch_rollout(
        self, instances: list["ProblemInstance"], n_rollouts: int,
    ) -> list[tuple[list[bool], list[dict]]]:
        """Batch rollout: 여러 문제를 한 번의 generate()로 처리."""
        from vllm import SamplingParams

        prompts = [_SOLVE_PROMPT.format(problem=inst.problem) for inst in instances]
        if not prompts:
            return []

        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n_rollouts,
        )
        all_outputs = self.llm.generate(prompts, params)

        results = []
        for out, inst in zip(all_outputs, instances):
            flags, logs = [], []
            for i, comp in enumerate(out.outputs):
                pred = _extract_boxed(comp.text)
                correct = _answers_match(pred, inst.answer) if pred else False
                flags.append(correct)
                logs.append({
                    "rollout_idx": i, "response": comp.text,
                    "extracted": pred, "ground_truth": inst.answer, "correct": correct,
                })
            results.append((flags, logs))
        return results


# ---------------------------------------------------------------------------
# Mock / Real LLM mutation
# ---------------------------------------------------------------------------

def _mock_mutate(parent: ProblemProgram, rng: random.Random) -> Optional[str]:
    """
    LLM 없이 소스 코드에 간단한 변형을 적용한다.
    새 program_id(MD5)를 생성하는 것이 목적이므로, 의미적 올바름보다
    코드 변형 가능성을 우선한다.

    우선순위:
      1. 소스 내 정수 리터럴 숫자 약간 조정 (가장 robust)
      2. 주석 삽입 (fallback)
    """
    import re as _re
    src = parent.source_code

    # 소스에 있는 모든 양의 정수 리터럴 목록
    int_positions = [
        m for m in _re.finditer(r'\b(\d+)\b', src)
        if 2 <= int(m.group()) <= 10000
    ]

    if int_positions:
        # 랜덤으로 하나 골라 ±10% 범위에서 조정
        target = rng.choice(int_positions)
        old_val = int(target.group())
        delta = max(1, int(old_val * 0.1))
        new_val = old_val + rng.choice([-delta, delta])
        new_val = max(1, new_val)
        if new_val == old_val:
            new_val += 1
        mutated = src[:target.start()] + str(new_val) + src[target.end():]
    else:
        # fallback: 고유 주석 삽입
        mutated = f"# variant-{rng.randint(10000, 99999)}\n" + src

    return mutated if mutated != src else None


def _llm_mutate(
    parent: ProblemProgram,
    model: str,
    base_url: Optional[str] = None,
    api_key: str = None,
    in_depth: bool = True,
) -> Optional[str]:
    """OpenAI-compatible API로 실제 mutation."""
    try:
        from openai import OpenAI
    except ImportError:
        print("[LLM] openai 패키지가 없습니다. pip install openai")
        return None

    prompt_tmpl = (
        "You are an expert mathematician and Python programmer.\n\n"
        "Below is a Python function that generates math problems using "
        "inverse construction (answer first, then problem):\n\n"
        "```python\n{code}\n```\n\n"
        "{instruction}\n\n"
        "RULES:\n"
        "1. Function MUST be named `generate` and take a single `seed` argument\n"
        "2. MUST return (problem_text: str, answer: str)\n"
        "3. answer MUST be constructed FIRST\n"
        "4. Use only standard library + math\n\n"
        "Return ONLY the Python code."
    )
    instruction = (
        "Make it HARDER (more steps, constraints, combined concepts)."
        if in_depth else
        "Create a COMPLETELY DIFFERENT math problem type."
    )
    prompt = prompt_tmpl.format(code=parent.source_code, instruction=instruction)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=10240,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[LLM] API call failed: {e}")
        return None

    # 코드 추출
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = m.group(1).strip() if m else text.strip()
    return code if "def generate" in code else None


# ---------------------------------------------------------------------------
# Mock rollout & entropy
# ---------------------------------------------------------------------------

def _mock_rollout(
    inst: ProblemInstance,
    n_rollouts: int,
    rng: random.Random,
    difficulty_bias: float = 0.5,
) -> list[bool]:
    """
    모델이 문제를 풀 확률을 랜덤으로 시뮬레이션.
    difficulty_bias ∈ (0, 1): 낮을수록 어렵게 설정.
    실제로는 veRL actor_rollout_wg.generate_sequences 호출.
    """
    p_true = rng.betavariate(2, 2)  # 0.2~0.8 사이 집중
    return [rng.random() < p_true for _ in range(n_rollouts)]


def _mock_entropy(
    inst: ProblemInstance,
    rng: random.Random,
    h_mean: float = 2.0,
    h_std: float = 0.8,
) -> float:
    """
    정규분포로 entropy를 시뮬레이션.
    실제로는 actor_rollout_wg.compute_log_prob(calculate_entropy=True).
    """
    h = rng.gauss(h_mean, h_std)
    return max(0.1, h)


# ---------------------------------------------------------------------------
# Single evolution step
# ---------------------------------------------------------------------------

def evolution_step(
    grid: MAPElitesGrid,
    candidates: int,
    n_rollouts: int,
    h_threshold: float,
    rng: random.Random,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: str = None,
    in_depth_ratio: float = 0.5,
    crossover_ratio: float = 0.2,
    verbose: bool = True,
    vllm_runner: Optional[VLLMRunner] = None,
) -> dict:
    """
    Evolution 1회 실행 — vLLM 사용 시 batched inference.

    3-phase pipeline:
      Phase 1: batch_mutate (1회 vLLM 호출) → execute (CPU)
      Phase 2: batch_rollout (1회 vLLM 호출)
      Phase 3: batch_entropy (1회 vLLM 호출)
      → 총 3회 GPU 호출로 전체 candidate 처리

    연산자 비율 (Mouret & Clune, 2015):
      crossover_ratio  → crossover (두 부모 결합)
      in_depth_ratio   → in-depth mutation (같은 유형 고도화)
      나머지            → in-breadth mutation (새 유형 탐색)
    """
    inserted = 0
    attempted = 0
    skipped_execute = 0
    skipped_h = 0
    candidate_logs = []

    # ================================================================
    # Phase 1: Mutation (batched if vllm_runner)
    # ================================================================
    mutation_tasks = []   # {"op", "parent", "parent_b"(optional)}
    for c in range(candidates):
        roll = rng.random()
        if roll < crossover_ratio:
            op = "crossover"
        elif roll < crossover_ratio + in_depth_ratio:
            op = "in_depth"
        else:
            op = "in_breadth"

        if op == "crossover":
            pa, pb = grid.sample_two_parents()
            if pa is None or pb is None:
                pa = grid.sample_parent()
                if pa is None:
                    continue
                op = "in_depth"
                mutation_tasks.append({"op": op, "parent": pa})
            else:
                mutation_tasks.append({"op": "crossover", "parent": pa, "parent_b": pb})
        else:
            parent = grid.sample_parent()
            if parent is None:
                continue
            mutation_tasks.append({"op": op, "parent": parent})

    # Batch mutation
    if vllm_runner and mutation_tasks:
        if verbose:
            print(f"  [batch] mutating {len(mutation_tasks)} candidates...")
        source_codes = vllm_runner.batch_mutate(mutation_tasks, grid=grid)
    else:
        source_codes = []
        for t in mutation_tasks:
            if t["op"] == "crossover":
                source_codes.append(_mock_mutate(t["parent"], rng))
            elif model:
                source_codes.append(
                    _llm_mutate(t["parent"], model, base_url, api_key, t["op"] == "in_depth"))
            else:
                source_codes.append(_mock_mutate(t["parent"], rng))

    # Build children + execute (CPU, fast)
    children = []       # (child, inst, task)
    for task, code in zip(mutation_tasks, source_codes):
        if code is None:
            if verbose:
                print(f"  {task['op']} mutation failed")
            continue

        if task["op"] == "crossover":
            pa, pb = task["parent"], task["parent_b"]
            child = ProblemProgram(
                source_code=code,
                parent_id=f"{pa.program_id}×{pb.program_id}",
                generation=max(pa.generation, pb.generation) + 1,
                metadata={"op": "crossover"},
            )
        else:
            child = ProblemProgram(
                source_code=code,
                parent_id=task["parent"].program_id,
                generation=task["parent"].generation + 1,
                metadata={"op": task["op"]},
            )

        # Multi-seed + SymPy 자가 검증
        inst = _verify_program(child, n_seeds=5)
        if inst is None:
            skipped_execute += 1
            if verbose:
                print(f"  execute failed ({task['op']})")
            continue
        children.append((child, inst, task))

    if not children:
        return {"attempted": 0, "inserted": 0, "skipped_execute": skipped_execute,
                "skipped_h": 0, "candidate_logs": []}

    # ================================================================
    # Phase 2: Rollout (batched — 1회 vLLM 호출)
    # ================================================================
    instances = [inst for _, inst, _ in children]
    if vllm_runner:
        if verbose:
            print(f"  [batch] rollout for {len(instances)} candidates...")
        all_rollout_results = vllm_runner.batch_rollout(instances, n_rollouts)
    else:
        all_rollout_results = [
            (_mock_rollout(inst, n_rollouts, rng), []) for inst in instances
        ]

    # ================================================================
    # Phase 3: Entropy (batched — 1회 vLLM 호출)
    # ================================================================
    if vllm_runner:
        if verbose:
            print(f"  [batch] entropy for {len(instances)} candidates...")
        all_h = vllm_runner.batch_entropy(instances)
    else:
        all_h = [_mock_entropy(inst, rng) for inst in instances]

    # ================================================================
    # Phase 4: Scoring + Grid insertion (CPU, fast)
    # ================================================================
    for (child, inst, task), (flags, rollout_logs), h_bar in zip(
        children, all_rollout_results, all_h
    ):
        attempted += 1
        p_hat = sum(flags) / len(flags) if flags else 0.0

        if h_bar is None:
            skipped_h += 1
            if verbose:
                print(f"  entropy failed, skip")
            continue

        if not h_prefilter(h_bar, h_threshold):
            skipped_h += 1
            if verbose:
                print(f"  H={h_bar:.3f} < threshold, skip")
            continue

        if not p_hat_filter(p_hat):
            skipped_h += 1
            if verbose:
                print(f"  p={p_hat:.2f} extreme (0 or 1), skip")
            continue

        rq_result = compute_rq_full(flags, h_bar)
        child.p_hat = p_hat
        child.h_score = h_bar
        child.rq_score = rq_result.rq_score
        child.fitness = rq_result.rq_score

        # 삽입 전 기존 챔피언 정보 저장 (before→after 비교용)
        target_h = grid.h_to_bin(h_bar)
        target_d = grid.problem_to_div_bin(inst.problem)
        old_niche = grid.grid.get((target_h, target_d))
        old_rq = old_niche.champion_rq if (old_niche and old_niche.champion) else -1
        old_problem = None
        if old_niche and old_niche.champion:
            old_inst = old_niche.champion.execute(seed=0)
            old_problem = old_inst.problem[:60] if old_inst else None

        was_inserted = grid.try_insert(
            program=child,
            h_value=h_bar,
            problem_text=inst.problem,
            rq_score=rq_result.rq_score,
        )

        # 상세 로그 기록
        candidate_logs.append({
            "candidate_idx": c,
            "op": task["op"],
            "problem": inst.problem,
            "answer": inst.answer,
            "p_hat": p_hat,
            "h_bar": h_bar,
            "rq_score": rq_result.rq_score,
            "inserted": was_inserted,
            "niche": (int(child.niche_h), int(child.niche_div)) if was_inserted else None,
            "generation": child.generation,
            "rollouts": rollout_logs,
        })

        if was_inserted:
            inserted += 1
            if verbose:
                op_label = task["op"][:5]
                if old_rq >= 0:
                    print(
                        f"  ✓ [{op_label}] REPLACED ({child.niche_h},{child.niche_div}) "
                        f"RQ {old_rq:.4f} → {rq_result.rq_score:.4f}  "
                        f"p={p_hat:.2f} H={h_bar:.2f}"
                    )
                    if old_problem:
                        print(f"    before: {old_problem}...")
                    print(f"    after:  {inst.problem[:60]}...")
                else:
                    print(
                        f"  ✓ [{op_label}] NEW ({child.niche_h},{child.niche_div}) "
                        f"RQ={rq_result.rq_score:.4f} p={p_hat:.2f} H={h_bar:.2f}"
                    )
                    print(f"    Q: {inst.problem[:60]}...")
        else:
            if verbose:
                print(
                    f"  ✗ not champion "
                    f"p={p_hat:.2f} H={h_bar:.2f} "
                    f"R_Q={rq_result.rq_score:.4f} "
                    f"(needed > {old_rq:.4f} at ({target_h},{target_d}))"
                )

    return {
        "attempted": attempted,
        "inserted": inserted,
        "skipped_execute": skipped_execute,
        "skipped_h": skipped_h,
        "candidate_logs": candidate_logs,
    }


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

def _snapshot_grid(grid: MAPElitesGrid) -> list[dict]:
    """현재 grid 상태를 직렬화 가능한 스냅샷으로 캡처."""
    import html as html_mod
    cells = []
    for d in range(grid.n_div_bins):
        for h in range(grid.n_h_bins):
            niche = grid.grid.get((h, d))
            rq = niche.champion_rq if (niche and niche.champion) else 0
            p = getattr(niche.champion, "p_hat", 0) if (niche and niche.champion) else 0
            h_val = getattr(niche.champion, "h_score", 0) if (niche and niche.champion) else 0
            gen = getattr(niche.champion, "generation", 0) if (niche and niche.champion) else 0
            has = 1 if (niche and niche.champion) else 0
            problem, answer = "", ""
            if niche and niche.champion:
                inst = niche.champion.execute(seed=0)
                if inst:
                    problem = inst.problem
                    answer = inst.answer
            cells.append({
                "h": h, "d": d, "rq": round(rq, 4), "p": round(p, 3),
                "H": round(h_val, 3), "gen": gen, "has": has,
                "problem": html_mod.escape(problem), "answer": html_mod.escape(answer),
            })
    return cells


def _save_html_dashboard(
    path: Path,
    grid: MAPElitesGrid,
    history: list[dict],
    champions_data: list[dict],
    seed_labels: dict[int, str],
    run_id: str,
    grid_snapshots: list[list[dict]] | None = None,
):
    """Self-contained HTML dashboard — step별 grid 비교 슬라이더 포함."""
    n_h = grid.n_h_bins
    n_d = grid.n_div_bins
    h_min, h_max = grid.h_range
    bw = (h_max - h_min) / n_h

    # 최신 grid (마지막 스냅샷 또는 직접 캡처)
    if grid_snapshots:
        all_snapshots = grid_snapshots
    else:
        all_snapshots = [_snapshot_grid(grid)]

    h_labels_json = json.dumps([f"H{i} [{h_min+i*bw:.1f}-{h_min+(i+1)*bw:.1f})" for i in range(n_h)])
    d_labels_json = json.dumps([seed_labels.get(d, f"D{d}") for d in range(n_d)])
    snapshots_json = json.dumps(all_snapshots)
    history_json = json.dumps(history)
    champs_json = json.dumps(champions_data, ensure_ascii=False, default=str)

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>RQ-Evolve Dashboard — {run_id}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
  h1 {{ color: #e94560; margin-bottom: 10px; }}
  h2 {{ color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 4px; margin: 20px 0 10px; }}
  .section {{ background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}

  .grid-container {{ overflow-x: auto; }}
  table.grid {{ border-collapse: collapse; }}
  table.grid th {{ background: #0f3460; padding: 6px 8px; font-size: 11px; position: sticky; top: 0; }}
  table.grid td {{ width: 80px; height: 40px; text-align: center; font-size: 11px;
                   border: 1px solid #1a1a2e; cursor: pointer; position: relative; transition: background 0.3s; }}
  table.grid td.empty {{ background: #0d1b2a; color: #333; }}
  table.grid td:hover {{ outline: 2px solid #e94560; z-index: 1; }}
  table.grid td.improved {{ box-shadow: inset 0 0 0 2px #00ff88; }}
  table.grid td.new-cell {{ box-shadow: inset 0 0 0 2px #ffd700; }}
  .row-label {{ background: #0f3460; padding: 6px 8px; font-size: 11px; text-align: right;
                white-space: nowrap; position: sticky; left: 0; z-index: 2; }}
  .tooltip {{ display: none; position: fixed; background: #16213e; border: 1px solid #e94560;
              border-radius: 6px; padding: 12px; max-width: 500px; z-index: 100; font-size: 12px;
              line-height: 1.5; }}
  .line {{ fill: none; stroke: #e94560; stroke-width: 2; }}
  .line-max {{ fill: none; stroke: #ffd700; stroke-width: 2; }}
  .line-hard {{ fill: none; stroke: #00ff88; stroke-width: 2; }}
  .slider-container {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .slider-container input {{ flex: 1; accent-color: #e94560; }}
  .slider-label {{ font-size: 18px; font-weight: bold; color: #e94560; min-width: 120px; }}
  .legend {{ display: flex; gap: 16px; margin-bottom: 8px; font-size: 11px; color: #aaa; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
  .legend .box {{ width: 14px; height: 14px; border-radius: 2px; }}
  .champ {{ background: #0d1b2a; border-radius: 6px; padding: 12px; margin: 8px 0; }}
  .champ .rq {{ color: #e94560; font-size: 18px; font-weight: bold; }}
  .champ .problem {{ color: #aaa; font-size: 12px; margin-top: 6px; }}
  .champ code {{ background: #1a1a2e; padding: 8px; display: block; font-size: 11px;
                 max-height: 200px; overflow-y: auto; margin-top: 6px; white-space: pre-wrap; }}
  .btn {{ background: #0f3460; color: #eee; border: 1px solid #e94560; padding: 4px 12px;
          border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .btn:hover {{ background: #e94560; }}
  .btn.active {{ background: #e94560; }}
</style>
</head>
<body>

<h1>RQ-Evolve Dashboard</h1>
<p style="color:#aaa">Run: {run_id}</p>

<h2>MAP-Elites Grid</h2>
<div class="section">
  <div class="slider-container">
    <span style="color:#aaa">Step:</span>
    <input type="range" id="stepSlider" min="0" max="0" value="0">
    <span class="slider-label" id="stepLabel">Init</span>
    <button class="btn" id="btnPlay">Play</button>
  </div>
  <div class="legend">
    <span><span class="box" style="background:#e94560"></span> R_Q 강도</span>
    <span><span class="box" style="border:2px solid #ffd700;background:transparent"></span> 새 셀 (이전 없음)</span>
    <span><span class="box" style="border:2px solid #00ff88;background:transparent"></span> 개선됨 (RQ 상승)</span>
  </div>
  <div class="grid-container">
    <table class="grid" id="gridTable"></table>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<h2>Evolution History</h2>
<div class="section">
  <div style="display:flex; gap:20px; flex-wrap:wrap;">
    <div style="flex:1; min-width:300px;">
      <p style="color:#aaa;">Mean R_Q (red) / Max R_Q (gold)</p>
      <svg id="chartRQ" viewBox="0 0 600 180"></svg>
    </div>
    <div style="flex:1; min-width:300px;">
      <p style="color:#aaa;">Coverage (red) / H2+ Champions (green)</p>
      <svg id="chartCov" viewBox="0 0 600 180"></svg>
    </div>
  </div>
</div>

<h2>Champions (by R_Q)</h2>
<div class="section" id="champList"></div>

<script>
const H_LABELS = {h_labels_json};
const D_LABELS = {d_labels_json};
const SNAPSHOTS = {snapshots_json};
const HISTORY = {history_json};
const CHAMPS = {champs_json};
const N_H = {n_h}, N_D = {n_d};

function rqColor(rq) {{
  if (rq <= 0) return '#0d1b2a';
  const t = Math.min(rq / 0.8, 1);
  const r = Math.round(35 + t * 198);
  const g = Math.round(27 + t * 42);
  const b = Math.round(46 + t * 50);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

// ── Grid rendering with step comparison ──
const slider = document.getElementById('stepSlider');
const label = document.getElementById('stepLabel');
slider.max = SNAPSHOTS.length - 1;
slider.value = SNAPSHOTS.length - 1;

function renderGrid(stepIdx) {{
  const grid = SNAPSHOTS[stepIdx];
  const prev = stepIdx > 0 ? SNAPSHOTS[stepIdx - 1] : null;
  label.textContent = stepIdx === 0 ? 'Init' : `Step ${{stepIdx}}`;

  const table = document.getElementById('gridTable');
  let html = '<tr><th></th>';
  H_LABELS.forEach(l => html += `<th>${{l}}</th>`);
  html += '</tr>';

  for (let d = 0; d < N_D; d++) {{
    html += `<tr><td class="row-label">${{D_LABELS[d]}}</td>`;
    for (let h = 0; h < N_H; h++) {{
      const cell = grid.find(c => c.h === h && c.d === d);
      const prevCell = prev ? prev.find(c => c.h === h && c.d === d) : null;

      if (cell && cell.has) {{
        let cls = '';
        let delta = '';
        if (prevCell) {{
          if (!prevCell.has) {{
            cls = 'new-cell';
            delta = ' (NEW)';
          }} else if (cell.rq > prevCell.rq + 0.001) {{
            cls = 'improved';
            delta = ` (+${{(cell.rq - prevCell.rq).toFixed(4)}})`;
          }}
        }}
        const info = `${{D_LABELS[d]}} | H${{h}}${{delta}}\\nRQ=${{cell.rq}} p=${{cell.p}} H=${{cell.H}}\\ngen=${{cell.gen}}\\nQ: ${{cell.problem}}\\nA: ${{cell.answer}}`;
        html += `<td class="${{cls}}" style="background:${{rqColor(cell.rq)}}" data-info="${{info}}">${{cell.rq.toFixed(3)}}</td>`;
      }} else {{
        html += '<td class="empty">·</td>';
      }}
    }}
    html += '</tr>';
  }}
  table.innerHTML = html;
}}

slider.addEventListener('input', () => renderGrid(+slider.value));
renderGrid(+slider.value);

// Play animation
let playing = false;
document.getElementById('btnPlay').addEventListener('click', function() {{
  if (playing) return;
  playing = true;
  this.classList.add('active');
  slider.value = 0;
  renderGrid(0);
  let i = 0;
  const iv = setInterval(() => {{
    i++;
    if (i >= SNAPSHOTS.length) {{ clearInterval(iv); playing = false; document.getElementById('btnPlay').classList.remove('active'); return; }}
    slider.value = i;
    renderGrid(i);
  }}, 800);
}});

// Tooltip
const tooltip = document.getElementById('tooltip');
document.getElementById('gridTable').addEventListener('mouseover', e => {{
  const info = e.target.getAttribute('data-info');
  if (info) {{ tooltip.innerHTML = info.replace(/\\\\n/g, '<br>').replace(/\\n/g, '<br>'); tooltip.style.display = 'block'; }}
}});
document.getElementById('gridTable').addEventListener('mousemove', e => {{
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 14) + 'px';
}});
document.getElementById('gridTable').addEventListener('mouseout', () => tooltip.style.display = 'none');

// ── Charts ──
function drawChart(svgId, data, lines) {{
  const svg = document.getElementById(svgId);
  if (!data.length) return;
  const W = 600, H = 160, P = 40;
  const n = data.length;
  const maxY = Math.max(...lines.flatMap(l => data.map(d => d[l.key] || 0)), 0.01) * 1.15;

  let html = `<line x1="${{P}}" y1="${{H}}" x2="${{W}}" y2="${{H}}" stroke="#333"/>`;
  html += `<text x="${{P-5}}" y="15" fill="#aaa" font-size="10" text-anchor="end">${{maxY.toFixed(2)}}</text>`;

  lines.forEach((l, li) => {{
    const pts = data.map((d, i) => `${{P + i*(W-P)/Math.max(n-1,1)}},${{H - (d[l.key]||0)/maxY*H}}`).join(' ');
    html += `<polyline points="${{pts}}" class="${{l.cls}}"/>`;
    html += `<text x="${{W-80}}" y="${{15 + li*15}}" fill="${{l.color}}" font-size="10">${{l.label}}</text>`;
    data.forEach((d, i) => {{
      const x = P + i*(W-P)/Math.max(n-1,1);
      html += `<circle cx="${{x}}" cy="${{H - (d[l.key]||0)/maxY*H}}" r="3" fill="${{l.color}}"/>`;
    }});
  }});
  svg.innerHTML = html;
}}

drawChart('chartRQ', HISTORY, [
  {{key:'mean_rq', cls:'line', color:'#e94560', label:'Mean R_Q'}},
  {{key:'max_rq', cls:'line-max', color:'#ffd700', label:'Max R_Q'}}
]);

// H2+ champion count from history (hard_champions field if available)
const covData = HISTORY.map((h, i) => ({{
  ...h,
  hard: h.hard_champions !== undefined ? h.hard_champions : 0
}}));
drawChart('chartCov', covData, [
  {{key:'coverage', cls:'line', color:'#e94560', label:'Coverage'}},
  {{key:'hard', cls:'line-hard', color:'#00ff88', label:'H2+ Champions'}}
]);

// ── Champions ──
const champDiv = document.getElementById('champList');
CHAMPS.forEach(c => {{
  const probs = (c.problems || []);
  let probHtml = probs.map(p => `<div class="problem"><b>seed=${{p.seed}}</b> Q: ${{p.problem}}<br>A: ${{p.answer}}</div>`).join('');
  champDiv.innerHTML += `
    <div class="champ">
      <span class="rq">RQ=${{(c.rq_score||0).toFixed(4)}}</span>
      &nbsp; p=${{(c.p_hat||0).toFixed(2)}} H=${{(c.h_score||0).toFixed(2)}} gen=${{c.generation}} niche=(${{c.niche_h}},${{c.niche_div}})
      ${{probHtml}}
      <code>${{c.source_code||''}}</code>
    </div>`;
}});
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ---------------------------------------------------------------------------
# Visualization helpers (terminal)
# ---------------------------------------------------------------------------

def print_grid(grid: MAPElitesGrid, seed_labels: dict[int, str] | None = None):
    """
    MAP-Elites grid 시각화.

    D축 라벨: seed_labels = {0: "equation", 1: "geometry", ...}
    셀 색상: RQ 값에 따라 시각적 강도 표시
    """
    print()

    # H축 bin 범위 표시
    h_min, h_max = grid.h_range
    bw = (h_max - h_min) / grid.n_h_bins
    h_labels = [f"H{i}" for i in range(grid.n_h_bins)]
    h_ranges = [f"[{h_min + i*bw:.1f}-{h_min + (i+1)*bw:.1f})" for i in range(grid.n_h_bins)]

    col_w = 9
    label_w = 18

    # 헤더
    header = " " * label_w + "".join(f"{l:>{col_w}}" for l in h_labels)
    ranges = " " * label_w + "".join(f"{r:>{col_w}}" for r in h_ranges)
    print(header)
    print(ranges)
    print(" " * label_w + "-" * (col_w * grid.n_h_bins))

    # Grid
    for d in range(grid.n_div_bins):
        label = seed_labels.get(d, f"D{d}")[:label_w - 4] if seed_labels else f"D{d}"
        row = f"{label:>{label_w - 2}} | "
        for h in range(grid.n_h_bins):
            niche = grid.grid.get((h, d))
            if niche and niche.champion is not None:
                rq = niche.champion_rq
                # RQ 강도 표시: ░ ▒ ▓ █
                if rq >= 0.5:
                    cell = f"█{rq:.3f}"
                elif rq >= 0.3:
                    cell = f"▓{rq:.3f}"
                elif rq >= 0.1:
                    cell = f"▒{rq:.3f}"
                else:
                    cell = f"░{rq:.3f}"
            else:
                cell = "   ·   "
            row += f"{cell:>{col_w}}"
        print(row)
    print()


def print_champion_detail(grid: MAPElitesGrid, seed_labels: dict[int, str] | None = None):
    """각 챔피언의 문제 예시를 보여줌."""
    champions = grid.get_all_champions()
    if not champions:
        return
    print("  ── Champion Details ──")
    for champ in sorted(champions, key=lambda c: -(c.rq_score or 0)):
        d_label = seed_labels.get(champ.niche_div, f"D{champ.niche_div}") if seed_labels else f"D{champ.niche_div}"
        inst = champ.execute(seed=0)
        if not inst:
            continue
        print(f"  [{champ.niche_h},{champ.niche_div}] {d_label:15s} "
              f"RQ={champ.rq_score:.4f} p={champ.p_hat:.2f} H={champ.h_score:.2f} gen={champ.generation}")
        print(f"    Q: {inst.problem[:90]}")
        print(f"    A: {inst.answer[:30]}")
    print()


def print_stats(stats: dict, label: str = ""):
    tag = f"[{label}] " if label else ""
    hard = stats.get('hard_champions', '?')
    print(
        f"  {tag}coverage={stats['coverage']:.0%}  "
        f"champions={stats['num_champions']}/{stats['total_niches']}  "
        f"H2+={hard}  "
        f"mean_rq={stats['mean_rq']:.4f}  "
        f"max_rq={stats['max_rq']:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evolution feasibility test")
    parser.add_argument("--seed_dir", default="./seed_programs")
    parser.add_argument("--n_evo", type=int, default=10, help="evolution steps (outer loop)")
    parser.add_argument("--candidates", type=int, default=8, help="batch size per round")
    parser.add_argument("--max_rounds", type=int, default=8,
                        help="fixed budget: rounds per evolution step (default: 8)")
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--n_h_bins", type=int, default=6)
    parser.add_argument("--n_div_bins", type=int, default=6)
    parser.add_argument("--h_range", type=float, nargs=2, default=[0.0, 5.0],
                        help="H축 범위 [min, max] (default: 0.0 5.0)")
    parser.add_argument("--h_threshold", type=float, default=0.1)
    parser.add_argument("--crossover_ratio", type=float, default=0.2,
                        help="crossover 연산자 비율 (default: 0.2)")
    parser.add_argument("--in_depth_ratio", type=float, default=0.5,
                        help="in-depth mutation 비율 (default: 0.5, 나머지=in-breadth)")
    parser.add_argument("--ucb_c", type=float, default=1.0,
                        help="UCB1 exploration coefficient (0 = greedy, higher = more exploration)")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="ε-greedy: uniform random selection probability (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42)
    # vLLM 실제 시나리오 (mutation + rollout + entropy 모두 실제 모델)
    parser.add_argument("--vllm_model", type=str, default=None,
                        help="vLLM 모델 경로 (e.g. Qwen/Qwen3-8B-Base). "
                             "지정 시 mutation/rollout/entropy 모두 실제 모델 사용.")
    parser.add_argument("--tp", type=int, default=1,
                        help="tensor_parallel_size (GPU 수)")
    parser.add_argument("--gpu_mem", type=float, default=0.85,
                        help="gpu_memory_utilization for vLLM")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=10240,
                        help="rollout/entropy 측정 시 최대 생성 토큰 수")
    # Optional OpenAI-compatible API (mutation만, entropy/rollout은 mock)
    parser.add_argument("--model", type=str, default=None,
                        help="OpenAI-compatible API 모델명 (e.g. gpt-4o-mini)")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./feasibility_out",
                        help="결과 저장 디렉터리")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    verbose = not args.quiet

    # ---- vLLM 초기화 (지정된 경우) ----
    vllm_runner: Optional[VLLMRunner] = None
    if args.vllm_model:
        vllm_runner = VLLMRunner(
            model_name=args.vllm_model,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    mode = (
        f"vLLM: {args.vllm_model}" if vllm_runner
        else f"API: {args.model}" if args.model
        else "MOCK"
    )
    rollout_note = "" if vllm_runner else "  (mock)"

    print("=" * 60)
    print("RQ-Evolve: Evolution Feasibility Test")
    print(f"  seed_dir   : {args.seed_dir}")
    print(f"  n_evo      : {args.n_evo}")
    print(f"  candidates : {args.candidates}")
    print(f"  n_rollouts : {args.n_rollouts}{rollout_note}")
    print(f"  grid       : {args.n_h_bins} × {args.n_div_bins}")
    print(f"  ucb_c      : {args.ucb_c}")
    print(f"  mode       : {mode}")
    print("=" * 60)

    # ---- 1. Seed programs ----
    print("\n[Phase 0] Loading seed programs...")
    seed_path = Path(args.seed_dir)
    seeds: list[ProblemProgram] = []
    for f in sorted(seed_path.glob("*.py")):
        prog = ProblemProgram(
            source_code=f.read_text(),
            generation=0,
            metadata={"source_file": f.name},
        )
        inst = prog.execute(seed=0, timeout=5.0)
        if inst:
            seeds.append(prog)
            if verbose:
                print(f"  ✓ {f.name:30s} → {inst.problem[:55]}...")
        else:
            print(f"  ✗ {f.name} (execute failed)")

    if not seeds:
        print("ERROR: no valid seeds found in", args.seed_dir)
        sys.exit(1)
    print(f"  Loaded {len(seeds)} seeds")

    # ---- 2. MAP-Elites init ----
    print("\n[Phase 0] Initializing MAP-Elites grid...")
    grid = MAPElitesGrid(
        n_h_bins=args.n_h_bins,
        n_div_bins=args.n_div_bins,
        h_range=tuple(args.h_range),
        ucb_c=args.ucb_c,
        epsilon=args.epsilon,
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

    seed_labels: dict[int, str] = {}
    for i in range(args.n_div_bins):
        seed_labels[i] = f"D{i}"
    print(f"  Grid: {args.n_h_bins} H-bins x {args.n_div_bins} D-bins (embedding-based)")

    # Insert seeds — vLLM 모드면 실제 entropy/rollout, 아니면 mock
    print(f"  Scoring {len(seeds)} seeds ({'vLLM' if vllm_runner else 'mock'})...")
    for prog in seeds:
        inst = prog.execute(seed=0)
        if not inst:
            continue
        if vllm_runner:
            h0 = vllm_runner.entropy(inst) or 1.0
            flags0, _ = vllm_runner.rollout(inst, args.n_rollouts)
        else:
            h0 = _mock_entropy(inst, rng, h_mean=1.5, h_std=0.5)
            flags0 = _mock_rollout(inst, args.n_rollouts, rng)
        rq0 = compute_rq_full(flags0, h0)
        prog.p_hat = rq0.p_hat
        prog.h_score = h0
        prog.rq_score = rq0.rq_score
        prog.fitness = rq0.rq_score
        grid.try_insert(
            program=prog,
            h_value=h0,
            problem_text=inst.problem,
            rq_score=rq0.rq_score,
        )
        if verbose:
            print(f"    {prog.metadata.get('source_file','?'):30s} "
                  f"H={h0:.2f}  p={rq0.p_hat:.2f}  RQ={rq0.rq_score:.4f}")

    print_stats(grid.stats(), "init")
    print_grid(grid, seed_labels)
    print_champion_detail(grid, seed_labels)

    # ---- 3. Evolution loop ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Evolution] Running {args.n_evo} evolution steps...\n")
    print(f"  Live dashboard: {out_dir}/dashboard_live.html\n")

    history = []
    all_candidate_logs = []  # 전체 rollout 상세 로그
    grid_snapshots = [_snapshot_grid(grid)]  # step 0 = 초기 상태
    total_inserted = 0
    total_attempted = 0

    for step in range(1, args.n_evo + 1):
        t0 = time.time()
        print(f"── Step {step}/{args.n_evo} " + "─" * 40)

        # Fixed-budget evolution: max_rounds per step (no early termination)
        step_attempted = 0
        step_inserted = 0

        for round_num in range(1, args.max_rounds + 1):
            round_result = evolution_step(
                grid=grid,
                candidates=args.candidates,
                n_rollouts=args.n_rollouts,
                h_threshold=args.h_threshold,
                rng=rng,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                in_depth_ratio=args.in_depth_ratio,
                crossover_ratio=args.crossover_ratio,
                verbose=verbose,
                vllm_runner=vllm_runner,
            )

            step_attempted += round_result["attempted"]
            step_inserted += round_result["inserted"]

            for log in round_result.get("candidate_logs", []):
                log["step"] = step
                log["round"] = round_num
                all_candidate_logs.append(log)

        total_inserted += step_inserted
        total_attempted += step_attempted

        stats = grid.stats()
        history.append({
            "step": step,
            "coverage": stats["coverage"],
            "mean_rq": stats["mean_rq"],
            "max_rq": stats["max_rq"],
            "champions": stats["num_champions"],
            "attempted": step_attempted,
            "inserted": step_inserted,
            "rounds": args.max_rounds,
            "skipped_execute": 0,
            "skipped_h": 0,
        })

        elapsed = time.time() - t0
        print_stats(stats, f"step {step}")
        print(
            f"  inserted={step_inserted}/{step_attempted}  "
            f"rounds={round_num}  "
            f"time={elapsed:.1f}s"
        )

        # 실시간 HTML dashboard 덮어쓰기 (매 step)
        _rt_champs = []
        for champ in sorted(grid.get_all_champions(), key=lambda c: -(c.rq_score or 0)):
            probs = []
            for s in range(5):
                inst_s = champ.execute(seed=s)
                if inst_s:
                    probs.append({"seed": s, "problem": inst_s.problem, "answer": inst_s.answer})
            _rt_champs.append({
                "program_id": champ.program_id,
                "generation": champ.generation,
                "niche_h": int(champ.niche_h) if hasattr(champ.niche_h, 'item') else champ.niche_h,
                "niche_div": int(champ.niche_div) if hasattr(champ.niche_div, 'item') else champ.niche_div,
                "rq_score": champ.rq_score, "p_hat": champ.p_hat,
                "h_score": champ.h_score, "source_code": champ.source_code,
                "problems": probs,
            })
        # Grid 스냅샷 추가
        grid_snapshots.append(_snapshot_grid(grid))

        _save_html_dashboard(
            out_dir / f"dashboard_live.html",
            grid, history, _rt_champs, seed_labels,
            f"live (step {step}/{args.n_evo})",
            grid_snapshots=grid_snapshots,
        )

    # ---- 4. Final report ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print_grid(grid, seed_labels)
    print_champion_detail(grid, seed_labels)
    print_stats(grid.stats(), "final")

    print(f"\n  Total attempted : {total_attempted}")
    print(f"  Total inserted  : {total_inserted}")
    insert_rate = total_inserted / total_attempted if total_attempted else 0
    print(f"  Insert rate     : {insert_rate:.1%}")

    print("\n  Evolution history (coverage, mean_rq, max_rq):")
    print(f"  {'Step':>6}  {'Coverage':>10}  {'Mean R_Q':>10}  {'Max R_Q':>10}  {'Champions':>10}")
    print("  " + "-" * 52)
    for h in history:
        print(
            f"  {h['step']:>6}  "
            f"{h['coverage']:>10.1%}  "
            f"{h['mean_rq']:>10.4f}  "
            f"{h['max_rq']:>10.4f}  "
            f"{h['champions']:>10}"
        )

    # ---- 5. Sample champion problems ----
    print("\n  Sample champion problems:")
    champions = grid.get_all_champions()
    shown = 0
    for champ in sorted(champions, key=lambda c: -c.rq_score)[:3]:
        inst = champ.execute(seed=0)
        if inst:
            print(f"\n  niche=({champ.niche_h},{champ.niche_div})  "
                  f"gen={champ.generation}  "
                  f"p_hat={champ.p_hat:.2f}  "
                  f"H={champ.h_score:.2f}  "
                  f"R_Q={champ.rq_score:.4f}")
            print(f"  Problem: {textwrap.shorten(inst.problem, 120)}")
            print(f"  Answer:  {inst.answer}")
            shown += 1

    # ---- 6. Feasibility verdict ----
    print("\n" + "=" * 60)
    print("FEASIBILITY CHECK")
    print("=" * 60)

    final_stats = grid.stats()
    init_coverage = history[0]["coverage"]
    final_coverage = history[-1]["coverage"]

    checks = {
        "Seed programs load & execute": len(seeds) > 0,
        "MAP-Elites grid initialized (has champions)": final_stats["num_champions"] > 0,
        "Evolution attempted mutations": total_attempted > 0,
        "Evolution inserted new champions": total_inserted > 0,
        "Grid coverage ≥ initial": final_coverage >= init_coverage,
    }

    all_pass = True
    for name, ok in checks.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  ✓ All checks passed — evolution pipeline is feasible!")
    else:
        print("  ✗ Some checks failed — review the output above.")
    print("=" * 60)

    # ---- 7. Save results ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # (a) evolution history JSON
    history_path = out_dir / f"history_{run_id}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # (b) champion problems JSON (전체, 문제 텍스트 포함)
    champions_data = []
    for champ in sorted(grid.get_all_champions(), key=lambda c: -c.rq_score):
        problems = []
        for s in range(5):
            inst = champ.execute(seed=s)
            if inst:
                problems.append({"seed": s, "problem": inst.problem, "answer": inst.answer})
        champions_data.append({
            "program_id": champ.program_id,
            "generation": champ.generation,
            "niche_h": champ.niche_h,
            "niche_div": champ.niche_div,
            "rq_score": champ.rq_score,
            "p_hat": champ.p_hat,
            "h_score": champ.h_score,
            "source_code": champ.source_code,
            "problems": problems,
        })
    champs_path = out_dir / f"champions_{run_id}.json"

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"):   # numpy scalar (int64, float32, ...)
                return obj.item()
            return super().default(obj)

    with open(champs_path, "w") as f:
        json.dump(champions_data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    # (c) grid heatmap CSV (R_Q per niche)
    csv_path = out_dir / f"grid_{run_id}.csv"
    with open(csv_path, "w") as f:
        f.write("div_bin," + ",".join(f"H{i}" for i in range(grid.n_h_bins)) + "\n")
        for d in range(grid.n_div_bins):
            row = [f"D{d}"]
            for h in range(grid.n_h_bins):
                niche = grid.grid.get((h, d))
                val = f"{niche.champion_rq:.4f}" if (niche and niche.champion) else ""
                row.append(val)
            f.write(",".join(row) + "\n")

    # (d) rollout 상세 로그 (문제 + 모델 응답 + 정답 비교)
    if all_candidate_logs:
        logs_path = out_dir / f"rollout_logs_{run_id}.json"
        with open(logs_path, "w") as f:
            json.dump(all_candidate_logs, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    # (e) HTML 시각화
    html_path = out_dir / f"dashboard_{run_id}.html"
    _save_html_dashboard(
        html_path, grid, history, champions_data, seed_labels, run_id,
        grid_snapshots=grid_snapshots,
    )

    print(f"\n  Results saved to: {out_dir}/")
    print(f"    history_{run_id}.json       — step별 coverage/rq 추이")
    print(f"    champions_{run_id}.json     — 챔피언 프로그램 + 문제 샘플")
    print(f"    grid_{run_id}.csv           — grid R_Q heatmap")
    print(f"    dashboard_{run_id}.html     — 시각화 대시보드 (브라우저에서 열기)")
    if all_candidate_logs:
        print(f"    rollout_logs_{run_id}.json  — 모델 응답 상세 로그")


if __name__ == "__main__":
    main()

# python scripts/test_feasibility.py --vllm_model Qwen/Qwen3-8B-Base  --tp 4 --n_evo 50 --n_h_bins 10 --n_div_bins 10
# python scripts/test_feasibility.py --model gpt-4o-mini --n_evo 10 --candidates 4  (API mutation only)