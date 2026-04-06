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
from rq_questioner.rq_score import compute_rq_full, h_prefilter
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Answer extraction helpers (rollout 정답 비교용)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def _extract_boxed(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def _answers_match(pred: str, gt: str) -> bool:
    if pred.strip().lower() == gt.strip().lower():
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# vLLM 실제 백엔드 (mutation / rollout / entropy)
# ---------------------------------------------------------------------------

_SOLVE_PROMPT = (
    "Solve the following math problem step by step.\n"
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)

# base 모델용 completion-style mutation 프롬프트
# "```python\nimport random\n" 까지 포함 → 모델이 이어서 완성
_MUTATE_DEPTH = (
    "# Python function that generates math word problems.\n"
    "# Rewrite to generate HARDER problems "
    "(more steps, combined concepts, multi-stage reasoning).\n\n"
    "# Original:\n"
    "```python\n{code}\n```\n\n"
    "# Harder version:\n"
    "```python\n"
    "import random\n"
)

_MUTATE_BREADTH = (
    "# Python function that generates math word problems.\n"
    "# Rewrite to generate a COMPLETELY DIFFERENT type of math problem "
    "(different topic, e.g. if original is geometry, try algebra or probability).\n\n"
    "# Original:\n"
    "```python\n{code}\n```\n\n"
    "# Different topic version:\n"
    "```python\n"
    "import random\n"
)


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
        max_tokens: int = 512,
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

    def entropy(self, inst: "ProblemInstance") -> Optional[float]:
        """
        문제에 대한 모델 응답 토큰의 평균 엔트로피 proxy를 반환.
        logprobs=1 → 각 위치의 top-1 log prob → H ≈ mean(-logprob).
        """
        from vllm import SamplingParams
        prompt = _SOLVE_PROMPT.format(problem=inst.problem)
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=1,
            n=1,
        )
        out = self.llm.generate([prompt], params)[0].outputs[0]
        if not out.logprobs:
            return None
        ents = [max(0.0, -next(iter(d.values())).logprob) for d in out.logprobs]
        return sum(ents) / len(ents) if ents else None

    # ---- rollout --------------------------------------------------------

    def rollout(self, inst: "ProblemInstance", n_rollouts: int) -> list[bool]:
        """G번 생성 → boxed 정답 추출 → 정오 판별."""
        from vllm import SamplingParams
        prompt = _SOLVE_PROMPT.format(problem=inst.problem)
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n_rollouts,
        )
        outputs = self.llm.generate([prompt], params)[0].outputs
        flags = []
        for comp in outputs:
            pred = _extract_boxed(comp.text)
            flags.append(_answers_match(pred, inst.answer) if pred else False)
        return flags

    # ---- mutation -------------------------------------------------------

    def mutate(self, parent: "ProblemProgram", in_depth: bool = True) -> Optional[str]:
        """
        Completion-style mutation (base 모델 호환).
        프롬프트 끝을 "```python\\nimport random\\n" 으로 끝내
        모델이 나머지 코드를 완성하도록 유도.
        """
        from vllm import SamplingParams
        tmpl = _MUTATE_DEPTH if in_depth else _MUTATE_BREADTH
        prompt = tmpl.format(code=parent.source_code)
        params = SamplingParams(
            temperature=0.1,   # 코드 생성: 낮은 temperature → syntax 오류 감소
            max_tokens=1024,
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
            max_tokens=1024,
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
    in_depth_ratio: float = 0.7,
    verbose: bool = True,
    vllm_runner: Optional[VLLMRunner] = None,
) -> dict:
    """
    Evolution 1회 실행.
    Returns: 통계 dict
    """
    inserted = 0
    attempted = 0
    skipped_execute = 0
    skipped_h = 0

    for c in range(candidates):
        parent = grid.sample_parent()
        if parent is None:
            print("  [warn] grid empty, skipping")
            continue

        # ---- Mutation ----
        in_depth = rng.random() < in_depth_ratio
        if vllm_runner:
            source_code = vllm_runner.mutate(parent, in_depth)
        elif model:
            source_code = _llm_mutate(parent, model, base_url, api_key, in_depth)
        else:
            source_code = _mock_mutate(parent, rng)

        if not source_code:
            if verbose:
                print(f"  [{c+1}/{candidates}] mutation failed")
            continue

        child = ProblemProgram(
            source_code=source_code,
            parent_id=parent.program_id,
            generation=parent.generation + 1,
            metadata={"in_depth": in_depth},
        )

        # ---- Execute ----
        inst = child.execute(seed=rng.randint(0, 9999), timeout=5.0)
        if inst is None:
            skipped_execute += 1
            if verbose:
                print(f"  [{c+1}/{candidates}] execute failed")
            continue

        attempted += 1

        # ---- Rollouts → p_hat ----
        if vllm_runner:
            flags = vllm_runner.rollout(inst, n_rollouts)
        else:
            flags = _mock_rollout(inst, n_rollouts, rng)
        p_hat = sum(flags) / len(flags)

        # ---- Entropy → H̄ ----
        if vllm_runner:
            h_bar = vllm_runner.entropy(inst)
            if h_bar is None:
                if verbose:
                    print(f"  [{c+1}/{candidates}] entropy failed, skip")
                skipped_h += 1
                continue
        else:
            h_bar = _mock_entropy(inst, rng)

        if not h_prefilter(h_bar, h_threshold):
            skipped_h += 1
            if verbose:
                print(f"  [{c+1}/{candidates}] H={h_bar:.3f} < threshold, skip")
            continue

        # ---- R_Q ----
        rq_result = compute_rq_full(flags, h_bar)
        child.p_hat = p_hat
        child.h_score = h_bar
        child.rq_score = rq_result.rq_score
        child.fitness = rq_result.rq_score

        was_inserted = grid.try_insert(
            program=child,
            h_value=h_bar,
            problem_text=inst.problem,
            rq_score=rq_result.rq_score,
        )

        if was_inserted:
            inserted += 1
            if verbose:
                print(
                    f"  [{c+1}/{candidates}] ✓ inserted "
                    f"p={p_hat:.2f} H={h_bar:.2f} "
                    f"R_Q={rq_result.rq_score:.4f} "
                    f"niche=({child.niche_h},{child.niche_div})"
                )
        else:
            if verbose:
                print(
                    f"  [{c+1}/{candidates}] ✗ not champion "
                    f"p={p_hat:.2f} H={h_bar:.2f} "
                    f"R_Q={rq_result.rq_score:.4f}"
                )

    return {
        "attempted": attempted,
        "inserted": inserted,
        "skipped_execute": skipped_execute,
        "skipped_h": skipped_h,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def print_grid(grid: MAPElitesGrid):
    """MAP-Elites grid 시각화."""
    print()
    h_labels = [f"H{i}" for i in range(grid.n_h_bins)]
    col_w = 8

    header = "      " + "".join(f"{l:>{col_w}}" for l in h_labels)
    print(header)
    print("      " + "-" * (col_w * grid.n_h_bins))

    for d in range(grid.n_div_bins):
        row = f"D{d}  | "
        for h in range(grid.n_h_bins):
            niche = grid.grid.get((h, d))
            if niche and niche.champion is not None:
                cell = f"{niche.champion_rq:.3f}"
            else:
                cell = "  ---  "
            row += f"{cell:>{col_w}}"
        print(row)
    print()


def print_stats(stats: dict, label: str = ""):
    tag = f"[{label}] " if label else ""
    print(
        f"  {tag}coverage={stats['coverage']:.0%}  "
        f"champions={stats['num_champions']}/{stats['total_niches']}  "
        f"mean_rq={stats['mean_rq']:.4f}  "
        f"max_rq={stats['max_rq']:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evolution feasibility test")
    parser.add_argument("--seed_dir", default="./seed_programs")
    parser.add_argument("--n_evo", type=int, default=10, help="evolution steps")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--n_h_bins", type=int, default=6)
    parser.add_argument("--n_div_bins", type=int, default=6)
    parser.add_argument("--h_threshold", type=float, default=0.1)
    parser.add_argument("--ucb_c", type=float, default=1.0,
                        help="UCB1 exploration coefficient (0 = greedy, higher = more exploration)")
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
    parser.add_argument("--max_tokens", type=int, default=512,
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
        h_range=(0.0, 5.0),
        ucb_c=args.ucb_c,
    )

    # Fit diversity axis from seed problems
    samples = []
    for prog in seeds:
        for s in range(5):
            inst = prog.execute(s)
            if inst:
                samples.append(inst.problem)
    if samples:
        grid.fit_diversity_axis(samples)
        print(f"  Diversity axis fitted ({len(samples)} samples)")

    # Insert seeds — vLLM 모드면 실제 entropy/rollout, 아니면 mock
    print(f"  Scoring {len(seeds)} seeds ({'vLLM' if vllm_runner else 'mock'})...")
    for prog in seeds:
        inst = prog.execute(seed=0)
        if not inst:
            continue
        if vllm_runner:
            h0 = vllm_runner.entropy(inst) or 1.0
            flags0 = vllm_runner.rollout(inst, args.n_rollouts)
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
    print_grid(grid)

    # ---- 3. Evolution loop ----
    print(f"[Evolution] Running {args.n_evo} evolution steps...\n")

    history = []
    total_inserted = 0
    total_attempted = 0

    for step in range(1, args.n_evo + 1):
        t0 = time.time()
        print(f"── Step {step}/{args.n_evo} " + "─" * 40)

        step_result = evolution_step(
            grid=grid,
            candidates=args.candidates,
            n_rollouts=args.n_rollouts,
            h_threshold=args.h_threshold,
            rng=rng,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            verbose=verbose,
            vllm_runner=vllm_runner,
        )

        total_inserted += step_result["inserted"]
        total_attempted += step_result["attempted"]

        stats = grid.stats()
        history.append({
            "step": step,
            "coverage": stats["coverage"],
            "mean_rq": stats["mean_rq"],
            "max_rq": stats["max_rq"],
            "champions": stats["num_champions"],
            **step_result,
        })

        elapsed = time.time() - t0
        print_stats(stats, f"step {step}")
        print(
            f"  inserted={step_result['inserted']}/{step_result['attempted']}  "
            f"exec_fail={step_result['skipped_execute']}  "
            f"h_fail={step_result['skipped_h']}  "
            f"time={elapsed:.1f}s"
        )

    # ---- 4. Final report ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print_grid(grid)
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

    print(f"\n  Results saved to: {out_dir}/")
    print(f"    history_{run_id}.json   — step별 coverage/rq 추이")
    print(f"    champions_{run_id}.json — 챔피언 프로그램 + 문제 샘플")
    print(f"    grid_{run_id}.csv       — grid R_Q heatmap")


if __name__ == "__main__":
    main()

# python scripts/test_feasibility.py --vllm_model Qwen/Qwen3-8B-Base --n_evo 10 --candidates 8 --n_rollouts 16
# python scripts/test_feasibility.py --vllm_model Qwen/Qwen3-8B-Base --tp 2 --n_evo 10 --candidates 8
# python scripts/test_feasibility.py --model gpt-4o-mini --n_evo 10 --candidates 4  (API mutation only)