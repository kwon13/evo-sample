"""
Evaluation module: benchmark evaluation following R-Zero methodology.

Benchmarks (from R-Zero paper):
  - GSM8K        (1319 test problems, pass@1)
  - MATH-500     (500 problems, pass@1)
  - AIME-2024    (30 problems, mean@32)

Evaluation uses the same vLLM engine as the pipeline's Phase 1.
Runs after each epoch to track Solver improvement.
"""

import re
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction & matching (shared with pipeline)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)
_HASH_RE = re.compile(r"####\s*(.+)")


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} or #### format."""
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    m = _HASH_RE.findall(text)
    if m:
        return m[-1].strip()
    return None


def normalize(s: str) -> str:
    s = s.strip().lower()
    for o, c in [("{", "}"), ("[", "]"), ("(", ")")]:
        if s.startswith(o) and s.endswith(c):
            s = s[1:-1].strip()
    s = s.rstrip(".").replace(",", "")
    return " ".join(s.split())


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


def answers_match(pred: str, gt: str) -> bool:
    if normalize(pred) == normalize(gt):
        return True
    result = _sympy_equal(pred, gt)
    if result is not None:
        return result
    return False


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


@dataclass
class BenchmarkProblem:
    problem: str
    answer: str
    source: str = ""
    difficulty: str = ""


def load_gsm8k(max_samples: int = -1) -> list[BenchmarkProblem]:
    """Load GSM8K test set from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        problems = []
        for i, item in enumerate(ds):
            if max_samples > 0 and i >= max_samples:
                break
            # Extract numeric answer after ####
            ans_match = re.search(r"####\s*(.+)", item["answer"])
            answer = ans_match.group(1).strip() if ans_match else ""
            problems.append(BenchmarkProblem(
                problem=item["question"],
                answer=answer,
                source="gsm8k",
            ))
        logger.info(f"Loaded GSM8K: {len(problems)} problems")
        return problems
    except Exception as e:
        logger.warning(f"Failed to load GSM8K: {e}")
        return []


def load_math500(max_samples: int = -1) -> list[BenchmarkProblem]:
    """Load MATH-500 from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = []
        for i, item in enumerate(ds):
            if max_samples > 0 and i >= max_samples:
                break
            problems.append(BenchmarkProblem(
                problem=item["problem"],
                answer=item["answer"],
                source="math500",
                difficulty=str(item.get("level", "")),
            ))
        logger.info(f"Loaded MATH-500: {len(problems)} problems")
        return problems
    except Exception as e:
        logger.warning(f"Failed to load MATH-500: {e}")
        return []


def load_aime2024(max_samples: int = -1) -> list[BenchmarkProblem]:
    """Load AIME 2024 from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        problems = []
        for i, item in enumerate(ds):
            if max_samples > 0 and i >= max_samples:
                break
            problems.append(BenchmarkProblem(
                problem=item["problem"],
                answer=str(item["answer"]),
                source="aime2024",
            ))
        logger.info(f"Loaded AIME-2024: {len(problems)} problems")
        return problems
    except Exception as e:
        logger.warning(f"Failed to load AIME-2024: {e}")
        return []


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    benchmark: str
    accuracy: float
    correct: int
    total: int
    metric: str           # "pass@1" or "mean@k"
    details: list = field(default_factory=list)


def evaluate_pass_at_1(
    llm,
    problems: list[BenchmarkProblem],
    benchmark_name: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> EvalResult:
    """
    Evaluate pass@1: generate one response per problem, check correctness.
    Uses greedy decoding (temperature=0) for deterministic results.
    """
    from vllm import SamplingParams

    # Build prompts
    prompts = []
    try:
        tok = llm.get_tokenizer()
        for p in problems:
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p.problem}],
                tokenize=False, add_generation_prompt=True,
            ))
    except Exception:
        for p in problems:
            prompts.append(f"{SYSTEM_PROMPT}\n\nProblem: {p.problem}\n\nSolution:")

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )

    outputs = llm.generate(prompts, params)

    correct = 0
    details = []
    for out, prob in zip(outputs, problems):
        response = out.outputs[0].text
        predicted = extract_answer(response)
        is_correct = (
            answers_match(predicted, prob.answer)
            if predicted else False
        )
        if is_correct:
            correct += 1
        details.append({
            "problem": prob.problem[:100],
            "gt": prob.answer,
            "pred": predicted or "",
            "correct": is_correct,
        })

    total = len(problems)
    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"  {benchmark_name}: {correct}/{total} = {accuracy:.1%}")

    return EvalResult(
        benchmark=benchmark_name,
        accuracy=accuracy,
        correct=correct,
        total=total,
        metric="pass@1",
        details=details,
    )


def evaluate_mean_at_k(
    llm,
    problems: list[BenchmarkProblem],
    benchmark_name: str,
    k: int = 32,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> EvalResult:
    """
    Evaluate mean@k (R-Zero style for AIME):
    Generate k responses per problem, count how many are correct,
    report average correctness across problems.
    """
    from vllm import SamplingParams

    prompts = []
    try:
        tok = llm.get_tokenizer()
        for p in problems:
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p.problem}],
                tokenize=False, add_generation_prompt=True,
            ))
    except Exception:
        for p in problems:
            prompts.append(f"{SYSTEM_PROMPT}\n\nProblem: {p.problem}\n\nSolution:")

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=k,
    )

    outputs = llm.generate(prompts, params)

    total_score = 0.0
    details = []
    for out, prob in zip(outputs, problems):
        n_correct = 0
        for comp in out.outputs:
            predicted = extract_answer(comp.text)
            if predicted and answers_match(predicted, prob.answer):
                n_correct += 1
        score = n_correct / k
        total_score += score
        details.append({
            "problem": prob.problem[:100],
            "gt": prob.answer,
            "n_correct": n_correct,
            "k": k,
            "score": score,
        })

    total = len(problems)
    accuracy = total_score / total if total > 0 else 0.0

    logger.info(f"  {benchmark_name}: mean@{k} = {accuracy:.1%}")

    return EvalResult(
        benchmark=benchmark_name,
        accuracy=accuracy,
        correct=int(total_score),
        total=total,
        metric=f"mean@{k}",
        details=details,
    )


# ---------------------------------------------------------------------------
# Main evaluation function (called by pipeline)
# ---------------------------------------------------------------------------

def run_evaluation(
    llm,
    epoch: int,
    gsm8k_samples: int = 200,
    math500_samples: int = 100,
    aime_k: int = 32,
    max_tokens: int = 2048,
) -> dict:
    """
    Run all benchmarks using the given vLLM engine.
    
    Called by the pipeline during Phase 1 (vLLM already loaded).
    Uses a subset of benchmarks for fast iteration; set samples=-1 for full eval.
    
    Returns dict of {benchmark_name: EvalResult}.
    """
    logger.info(f"[Eval] Running benchmark evaluation (epoch {epoch + 1})")
    results = {}

    # GSM8K — pass@1
    gsm8k = load_gsm8k(max_samples=gsm8k_samples)
    if gsm8k:
        results["gsm8k"] = evaluate_pass_at_1(
            llm, gsm8k, "GSM8K", max_tokens=max_tokens
        )

    # MATH-500 — pass@1
    math500 = load_math500(max_samples=math500_samples)
    if math500:
        results["math500"] = evaluate_pass_at_1(
            llm, math500, "MATH-500", max_tokens=max_tokens
        )

    # AIME-2024 — mean@k
    aime = load_aime2024()
    if aime:
        results["aime2024"] = evaluate_mean_at_k(
            llm, aime, "AIME-2024", k=aime_k, max_tokens=max_tokens
        )

    # Summary
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "accuracy": res.accuracy,
            "metric": res.metric,
            "correct": res.correct,
            "total": res.total,
        }
    logger.info(f"[Eval] Summary: {json.dumps(summary, indent=2)}")

    return results
