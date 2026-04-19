"""
Evolution Feasibility Test (Ollama backend)

test_feasibility.py의 로직(seed 로드 → MAP-Elites → evolution loop → dashboard)
을 그대로 유지하면서, VLLMRunner 대신 로컬에 실행 중인 Ollama 서버를 사용해
mutation / rollout / entropy 를 수행한다.

사용 전 준비:
  1) `ollama serve` 가 실행 중이어야 한다 (기본 http://localhost:11434)
  2) `ollama pull <모델명>` 로 대상 모델을 미리 받아 둔다
  3) `pip install ollama`  (Python 클라이언트)

사용 예:
  # 기본 실행
  python scripts/test_feasibility_ollama.py \
      --ollama_model qwen2.5:7b \
      --n_evo 10 --candidates 8

  # 원격/다른 호스트
  python scripts/test_feasibility_ollama.py \
      --ollama_model llama3.1:8b \
      --ollama_host http://127.0.0.1:11434

주의 — entropy 측정 방식의 차이:
  vLLM 버전은 per-token top-K logprobs 로 Shannon entropy 를 근사하지만,
  Ollama API 는 per-token logprobs 를 노출하지 않는다. 따라서 이 파일에서는
  "의미론적 엔트로피(semantic entropy)"를 사용한다:
     1) temperature=1.0 으로 N 회 샘플링
     2) 각 샘플에서 \\boxed{...} 답을 추출
     3) 답 분포 p(a)=count(a)/N 로부터 H = -Σ p(a) log p(a)
  → 모델이 자신 있게 같은 답을 내면 H≈0, 매번 다른 답을 내면 H≈log(N).
  vLLM 버전과 스케일이 다르므로 --h_range / --h_threshold 를 조정해야 할 수 있다.
"""

import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import math
import random
import sys
import time
import textwrap
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

# 프로젝트 루트 + scripts/ 를 path 에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from rq_questioner.program import ProblemProgram, ProblemInstance
from rq_questioner.map_elites import MAPElitesGrid
from rq_questioner.rq_score import compute_rq_full
from rq_questioner.code_utils import extract_generator_code
from prompts import (
    MUTATE_DEPTH, MUTATE_BREADTH, MUTATE_CROSSOVER,
    SCORE_FEEDBACK,
    score_diagnosis, build_few_shot_examples, build_execution_feedback,
    SOLVER_COMPLETION_PROMPT,
)
from dotenv import load_dotenv
load_dotenv()

# test_feasibility.py 에 이미 구현된 공용 헬퍼를 그대로 재사용한다.
# evolution_step() 은 내부에서 vllm_runner.batch_mutate / batch_rollout /
# batch_entropy 를 호출하므로, 동일한 인터페이스를 가진 OllamaRunner 를
# 그대로 주입(duck typing)하면 로직을 수정하지 않고 Ollama 로 동작한다.
from test_feasibility import (
    _extract_boxed,
    _answers_match,
    evolution_step,
    _snapshot_grid,
    _save_html_dashboard,
    print_grid,
    print_champion_detail,
    print_stats,
)

_SOLVE_PROMPT = SOLVER_COMPLETION_PROMPT


# ---------------------------------------------------------------------------
# Ollama runner — VLLMRunner 와 동일한 메서드 시그니처
# ---------------------------------------------------------------------------

class OllamaRunner:
    """
    Ollama 기반 mutation / rollout / entropy.

    VLLMRunner 와 동일한 공개 인터페이스를 가진다:
        entropy(inst, top_k=...)                      -> float | None
        rollout(inst, n_rollouts)                     -> (flags, logs)
        mutate(parent, in_depth, grid)                -> code | None
        crossover(parent_a, parent_b, grid)           -> code | None
        batch_mutate(tasks, grid)                     -> list[code | None]
        batch_rollout(instances, n_rollouts)          -> list[(flags, logs)]
        batch_entropy(instances, top_k=...)           -> list[float | None]

    entropy 는 의미론적 엔트로피(semantic entropy) 로 근사한다
    (자세한 내용은 모듈 docstring 참조).
    """

    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        n_entropy_samples: int = 2,
        max_parallel: int = 4,
    ):
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "`ollama` 패키지가 설치되어 있지 않습니다. "
                "`pip install ollama` 로 설치 후 다시 실행하세요."
            ) from e

        print(f"[Ollama] Connecting to {host} ...")
        self.client = ollama.Client(host=host)

        # 연결 & 모델 존재 여부 확인 (없으면 경고만)
        try:
            resp = self.client.list()
        except Exception as e:
            raise RuntimeError(
                f"[Ollama] 서버에 연결할 수 없습니다 ({host}): {e}\n"
                "  → `ollama serve` 가 실행 중인지 확인하세요."
            ) from e

        available = _extract_model_names(resp)
        if available and model_name not in available:
            print(
                f"[Ollama] 경고: '{model_name}' 이(가) 설치된 모델 목록에 없습니다.\n"
                f"         설치된 모델: {available}\n"
                f"         필요 시 `ollama pull {model_name}` 로 먼저 받아 두세요."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_entropy_samples = max(2, int(n_entropy_samples))
        self.max_parallel = max(1, int(max_parallel))
        print(
            f"[Ollama] Ready — model={model_name}, "
            f"max_parallel={self.max_parallel}, "
            f"n_entropy_samples={self.n_entropy_samples}"
        )

    # ---- 단일/병렬 completion ------------------------------------------------

    def _complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Ollama 에 단일 completion 요청."""
        try:
            resp = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens),
                },
                stream=False,
            )
        except Exception as e:
            print(f"[Ollama] generate 실패: {e}")
            return ""
        return _extract_response_text(resp)

    def _complete_many(
        self, prompts: list[str], temperature: float, max_tokens: int,
    ) -> list[str]:
        """여러 prompt 를 스레드 풀로 병렬 호출."""
        if not prompts:
            return []
        if self.max_parallel == 1 or len(prompts) == 1:
            return [self._complete(p, temperature, max_tokens) for p in prompts]
        with ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
            return list(
                ex.map(
                    lambda p: self._complete(p, temperature, max_tokens),
                    prompts,
                )
            )

    # ---- entropy (semantic) -------------------------------------------------

    def entropy(self, inst: "ProblemInstance", top_k: int = 20) -> Optional[float]:
        """
        temperature=1.0 으로 N 회 샘플링 → 답 분포의 Shannon entropy.
        `top_k` 인자는 VLLMRunner 인터페이스 호환을 위해 받되 사용하지 않는다.
        """
        prompt = _SOLVE_PROMPT.format(problem=inst.problem)
        texts = self._complete_many(
            [prompt] * self.n_entropy_samples,
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return _semantic_entropy_from_texts(texts)

    def batch_entropy(
        self, instances: list["ProblemInstance"], top_k: int = 20,
    ) -> list[Optional[float]]:
        """
        여러 instance 에 대한 entropy 를 batched 로 측정.

        (n_instances × n_entropy_samples) 개의 prompt 를 하나의 스레드 풀에서
        병렬 처리하므로 instance 별 개별 호출보다 훨씬 효율적이다.
        """
        if not instances:
            return []
        flat_prompts: list[str] = []
        for inst in instances:
            p = _SOLVE_PROMPT.format(problem=inst.problem)
            flat_prompts.extend([p] * self.n_entropy_samples)

        texts = self._complete_many(
            flat_prompts,
            temperature=1.0,
            max_tokens=min(self.max_tokens, 512),
        )

        n = self.n_entropy_samples
        results: list[Optional[float]] = []
        for i in range(len(instances)):
            chunk = texts[i * n : (i + 1) * n]
            results.append(_semantic_entropy_from_texts(chunk))
        return results

    # ---- rollout ------------------------------------------------------------

    def rollout(
        self, inst: "ProblemInstance", n_rollouts: int,
    ) -> tuple[list[bool], list[dict]]:
        prompt = _SOLVE_PROMPT.format(problem=inst.problem)
        texts = self._complete_many(
            [prompt] * n_rollouts,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _score_rollout_texts(texts, inst)

    def batch_rollout(
        self, instances: list["ProblemInstance"], n_rollouts: int,
    ) -> list[tuple[list[bool], list[dict]]]:
        if not instances:
            return []
        flat_prompts: list[str] = []
        for inst in instances:
            p = _SOLVE_PROMPT.format(problem=inst.problem)
            flat_prompts.extend([p] * n_rollouts)

        texts = self._complete_many(
            flat_prompts,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        results = []
        for i, inst in enumerate(instances):
            chunk = texts[i * n_rollouts : (i + 1) * n_rollouts]
            results.append(_score_rollout_texts(chunk, inst))
        return results

    # ---- mutation -----------------------------------------------------------

    def _extract_code(self, text: str) -> Optional[str]:
        """Mutation/crossover 출력에서 코드 블록 추출 (VLLMRunner 와 동일)."""
        return extract_generator_code(text)

    def mutate(
        self, parent: "ProblemProgram", in_depth: bool = True,
        grid: Optional["MAPElitesGrid"] = None,
    ) -> Optional[str]:
        prompt = _build_mutate_prompt(parent, in_depth=in_depth, grid=grid)
        text = self._complete(prompt, temperature=0.1, max_tokens=4096)
        return self._extract_code(text)

    def crossover(
        self, parent_a: "ProblemProgram", parent_b: "ProblemProgram",
        grid: Optional["MAPElitesGrid"] = None,
    ) -> Optional[str]:
        prompt = _build_crossover_prompt(parent_a, parent_b, grid=grid)
        text = self._complete(prompt, temperature=0.3, max_tokens=4096)
        return self._extract_code(text)

    def batch_mutate(
        self, tasks: list[dict], grid: Optional["MAPElitesGrid"] = None,
    ) -> list[Optional[str]]:
        if not tasks:
            return []
        prompts: list[str] = []
        for t in tasks:
            if t["op"] == "crossover":
                prompts.append(
                    _build_crossover_prompt(t["parent"], t["parent_b"], grid=grid)
                )
            else:
                prompts.append(
                    _build_mutate_prompt(
                        t["parent"], in_depth=(t["op"] == "in_depth"), grid=grid,
                    )
                )

        texts = self._complete_many(prompts, temperature=0.3, max_tokens=4096)
        return [self._extract_code(t) for t in texts]


# ---------------------------------------------------------------------------
# Prompt / parsing helpers
# ---------------------------------------------------------------------------

def _build_mutate_prompt(
    parent: "ProblemProgram", in_depth: bool,
    grid: Optional["MAPElitesGrid"],
) -> str:
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
    return tmpl.format(
        code=parent.source_code, score_feedback=score_feedback,
        few_shot=few_shot, exec_feedback=exec_fb,
    )


def _build_crossover_prompt(
    parent_a: "ProblemProgram", parent_b: "ProblemProgram",
    grid: Optional["MAPElitesGrid"],
) -> str:
    few_shot = build_few_shot_examples(grid) if grid else ""
    return MUTATE_CROSSOVER.format(
        code_a=parent_a.source_code, code_b=parent_b.source_code,
        p_hat_a=getattr(parent_a, "p_hat", 0.5),
        h_a=getattr(parent_a, "h_score", 1.0),
        p_hat_b=getattr(parent_b, "p_hat", 0.5),
        h_b=getattr(parent_b, "h_score", 1.0),
        few_shot=few_shot,
    )


def _score_rollout_texts(
    texts: list[str], inst: "ProblemInstance",
) -> tuple[list[bool], list[dict]]:
    flags: list[bool] = []
    logs: list[dict] = []
    for i, text in enumerate(texts):
        pred = _extract_boxed(text) if text else None
        correct = _answers_match(pred, inst.answer) if pred else False
        flags.append(correct)
        logs.append({
            "rollout_idx": i,
            "response": text,
            "extracted": pred,
            "ground_truth": inst.answer,
            "correct": correct,
        })
    return flags, logs


def _semantic_entropy_from_texts(texts: list[str]) -> Optional[float]:
    """
    샘플 텍스트 리스트에서 의미론적 엔트로피 계산.

      1) \\boxed{...} 답 추출 (없으면 텍스트 끝부분 80자를 signature 로 사용)
      2) Counter 로 빈도 분포 구성
      3) H = -Σ (c/N) log (c/N)

    모든 샘플이 동일 답 → H=0.  모든 샘플이 서로 다른 답 → H=log(N).
    """
    answers: list[str] = []
    for t in texts:
        if not t:
            continue
        pred = _extract_boxed(t)
        if pred is None:
            pred = t.strip()[-80:]
        answers.append(pred.strip())

    if not answers:
        return None

    counts = Counter(answers)
    total = sum(counts.values())
    return -sum(
        (c / total) * math.log(c / total)
        for c in counts.values() if c > 0
    )


def _extract_response_text(resp) -> str:
    """Ollama 응답 객체(dict/pydantic) 에서 본문 텍스트 추출."""
    if resp is None:
        return ""
    text = getattr(resp, "response", None)
    if text is not None:
        return text or ""
    if isinstance(resp, dict):
        return resp.get("response", "") or ""
    try:
        return resp["response"] or ""
    except Exception:
        return ""


def _extract_model_names(resp) -> list[str]:
    """`client.list()` 결과에서 설치된 모델 이름 목록을 얻는다."""
    names: list[str] = []
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    for m in models or []:
        name = (
            getattr(m, "model", None)
            or getattr(m, "name", None)
            or (m.get("model") if isinstance(m, dict) else None)
            or (m.get("name") if isinstance(m, dict) else None)
        )
        if name:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evolution feasibility test (Ollama backend)"
    )
    parser.add_argument("--seed_dir", default="./seed_programs")
    parser.add_argument("--n_evo", type=int, default=10, help="evolution steps")
    parser.add_argument("--candidates", type=int, default=8, help="candidates per round")
    parser.add_argument("--max_rounds", type=int, default=8,
                        help="fixed budget: rounds per evolution step")
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--n_h_bins", type=int, default=6)
    parser.add_argument("--n_div_bins", type=int, default=6)
    parser.add_argument("--h_range", type=float, nargs=2, default=[0.0, 2.5],
                        help="H축 범위 [min, max] — semantic entropy 는 vLLM 보다 "
                             "스케일이 작으므로 기본값을 [0, 2.5] 로 설정")
    parser.add_argument("--h_threshold", type=float, default=0.05)
    parser.add_argument("--crossover_ratio", type=float, default=0.2)
    parser.add_argument("--in_depth_ratio", type=float, default=0.5)
    parser.add_argument("--ucb_c", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    # Ollama 관련
    parser.add_argument("--ollama_model", type=str, required=True,
                        help="Ollama 모델명 (예: qwen2.5:7b, llama3.1:8b)")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--n_entropy_samples", type=int, default=8,
                        help="semantic entropy 추정용 샘플 수 (클수록 안정적)")
    parser.add_argument("--max_parallel", type=int, default=4,
                        help="Ollama 에 동시 전송할 요청 수")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./feasibility_out")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    verbose = not args.quiet

    # ---- Ollama 초기화 ----
    runner = OllamaRunner(
        model_name=args.ollama_model,
        host=args.ollama_host,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n_entropy_samples=args.n_entropy_samples,
        max_parallel=args.max_parallel,
    )

    mode = f"Ollama: {args.ollama_model} @ {args.ollama_host}"

    print("=" * 60)
    print("RQ-Evolve: Evolution Feasibility Test (Ollama)")
    print(f"  seed_dir   : {args.seed_dir}")
    print(f"  n_evo      : {args.n_evo}")
    print(f"  candidates : {args.candidates}")
    print(f"  n_rollouts : {args.n_rollouts}")
    print(f"  grid       : {args.n_h_bins} × {args.n_div_bins}")
    print(f"  h_range    : [{args.h_range[0]}, {args.h_range[1]}]")
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

    seed_problems = []
    for prog in seeds:
        for s in range(5):
            inst = prog.execute(seed=s)
            if inst:
                seed_problems.append(inst.problem)
    if seed_problems:
        grid.fit_diversity_axis(seed_problems)
        print(f"  D-axis fitted with {len(seed_problems)} seed problems")

    seed_labels: dict[int, str] = {i: f"D{i}" for i in range(args.n_div_bins)}
    print(f"  Grid: {args.n_h_bins} H-bins x {args.n_div_bins} D-bins (embedding-based)")

    # Insert seeds — batched entropy/rollout 으로 Ollama 호출 횟수를 줄인다.
    print(f"  Scoring {len(seeds)} seeds (Ollama, batched)...")
    seed_instances: list[ProblemInstance] = []
    valid_progs: list[ProblemProgram] = []
    for prog in seeds:
        inst = prog.execute(seed=0)
        if inst:
            seed_instances.append(inst)
            valid_progs.append(prog)

    all_h = runner.batch_entropy(seed_instances)
    all_rollouts = runner.batch_rollout(seed_instances, args.n_rollouts)

    for prog, inst, h0, (flags0, _) in zip(
        valid_progs, seed_instances, all_h, all_rollouts
    ):
        h0 = h0 if h0 is not None else 1.0
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
            print(
                f"    {prog.metadata.get('source_file','?'):30s} "
                f"H={h0:.2f}  p={rq0.p_hat:.2f}  RQ={rq0.rq_score:.4f}"
            )

    print_stats(grid.stats(), "init")
    print_grid(grid, seed_labels)
    print_champion_detail(grid, seed_labels)

    # ---- 3. Evolution loop ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Evolution] Running {args.n_evo} evolution steps...\n")
    print(f"  Live dashboard: {out_dir}/dashboard_live_ollama.html\n")

    history: list[dict] = []
    all_candidate_logs: list[dict] = []
    grid_snapshots = [_snapshot_grid(grid)]
    total_inserted = 0
    total_attempted = 0

    for step in range(1, args.n_evo + 1):
        t0 = time.time()
        print(f"── Step {step}/{args.n_evo} " + "─" * 40)

        step_attempted = 0
        step_inserted = 0
        round_num = 0

        for round_num in range(1, args.max_rounds + 1):
            # OllamaRunner 는 VLLMRunner 와 동일한 batch_* 인터페이스를 가지므로
            # `vllm_runner=runner` 로 그대로 주입할 수 있다 (duck typing).
            round_result = evolution_step(
                grid=grid,
                candidates=args.candidates,
                n_rollouts=args.n_rollouts,
                h_threshold=args.h_threshold,
                rng=rng,
                in_depth_ratio=args.in_depth_ratio,
                crossover_ratio=args.crossover_ratio,
                verbose=verbose,
                vllm_runner=runner,
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

        # Live dashboard 갱신
        _rt_champs = []
        for champ in sorted(grid.get_all_champions(), key=lambda c: -(c.rq_score or 0)):
            probs = []
            for s in range(5):
                inst_s = champ.execute(seed=s)
                if inst_s:
                    probs.append({
                        "seed": s,
                        "problem": inst_s.problem,
                        "answer": inst_s.answer,
                    })
            _rt_champs.append({
                "program_id": champ.program_id,
                "generation": champ.generation,
                "niche_h": int(champ.niche_h) if hasattr(champ.niche_h, "item") else champ.niche_h,
                "niche_div": int(champ.niche_div) if hasattr(champ.niche_div, "item") else champ.niche_div,
                "rq_score": champ.rq_score, "p_hat": champ.p_hat,
                "h_score": champ.h_score, "source_code": champ.source_code,
                "problems": probs,
            })
        grid_snapshots.append(_snapshot_grid(grid))

        _save_html_dashboard(
            out_dir / "dashboard_live_ollama.html",
            grid, history, _rt_champs, seed_labels,
            f"ollama-live (step {step}/{args.n_evo})",
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
    for champ in sorted(champions, key=lambda c: -c.rq_score)[:3]:
        inst = champ.execute(seed=0)
        if inst:
            print(
                f"\n  niche=({champ.niche_h},{champ.niche_div})  "
                f"gen={champ.generation}  "
                f"p_hat={champ.p_hat:.2f}  "
                f"H={champ.h_score:.2f}  "
                f"R_Q={champ.rq_score:.4f}"
            )
            print(f"  Problem: {textwrap.shorten(inst.problem, 120)}")
            print(f"  Answer:  {inst.answer}")

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
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            return super().default(obj)

    # (a) history
    history_path = out_dir / f"history_ollama_{run_id}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # (b) champions
    champions_data = []
    for champ in sorted(grid.get_all_champions(), key=lambda c: -c.rq_score):
        problems = []
        for s in range(5):
            inst = champ.execute(seed=s)
            if inst:
                problems.append({
                    "seed": s, "problem": inst.problem, "answer": inst.answer,
                })
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
    champs_path = out_dir / f"champions_ollama_{run_id}.json"
    with open(champs_path, "w") as f:
        json.dump(champions_data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    # (c) grid CSV
    csv_path = out_dir / f"grid_ollama_{run_id}.csv"
    with open(csv_path, "w") as f:
        f.write("div_bin," + ",".join(f"H{i}" for i in range(grid.n_h_bins)) + "\n")
        for d in range(grid.n_div_bins):
            row = [f"D{d}"]
            for h in range(grid.n_h_bins):
                niche = grid.grid.get((h, d))
                val = f"{niche.champion_rq:.4f}" if (niche and niche.champion) else ""
                row.append(val)
            f.write(",".join(row) + "\n")

    # (d) rollout logs
    if all_candidate_logs:
        logs_path = out_dir / f"rollout_logs_ollama_{run_id}.json"
        with open(logs_path, "w") as f:
            json.dump(
                all_candidate_logs, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder,
            )

    # (e) HTML dashboard
    html_path = out_dir / f"dashboard_ollama_{run_id}.html"
    _save_html_dashboard(
        html_path, grid, history, champions_data, seed_labels,
        f"ollama_{run_id}",
        grid_snapshots=grid_snapshots,
    )

    print(f"\n  Results saved to: {out_dir}/")
    print(f"    history_ollama_{run_id}.json       — step 별 coverage/rq 추이")
    print(f"    champions_ollama_{run_id}.json     — 챔피언 프로그램 + 샘플")
    print(f"    grid_ollama_{run_id}.csv           — grid R_Q heatmap")
    print(f"    dashboard_ollama_{run_id}.html     — 시각화 대시보드")
    if all_candidate_logs:
        print(f"    rollout_logs_ollama_{run_id}.json  — 모델 응답 상세 로그")


if __name__ == "__main__":
    main()

# 예시:
# python scripts/test_feasibility_ollama.py --ollama_model qwen2.5:7b --n_evo 5 --candidates 4
# python scripts/test_feasibility_ollama.py --ollama_model llama3.1:8b --max_parallel 8 --n_evo 10
