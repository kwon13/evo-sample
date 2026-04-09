"""
Evolutionary Pipeline: Algorithm 4.1 — Self-Evolving single-model architecture.

One model plays both roles:
  - Questioner: mutates problem-generation programs (LLM writes Python)
  - Solver: solves the generated problems (LLM does math reasoning)

Phase-separated execution per epoch:
  Phase 1  (vLLM loaded) — Questioner evolution
    1-2.  Sample parent + LLM mutate program
    3.    Execute program → natural language (problem, answer)
    4.    Verify via SymPy / majority-vote
    5.    H pre-filter: vLLM (n=1, logprobs) → entropy
    6.    G rollouts: vLLM (n=G) → p_hat
    7.    R_Q = p(1-p) · H → MAP-Elites update
    → Destroy vLLM (free GPU)

  Phase 2  (veRL subprocess) — Solver GRPO training
    8-9.  Champions → fresh problem instances → parquet
    10.   veRL GRPO trains the SAME model on evolved problems
    → Checkpoint saved

  Next epoch: Phase 1 loads checkpoint → better mutator + better solver
"""

import os
import re
import gc
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .program import ProblemProgram, ProblemInstance
from .map_elites import MAPElitesGrid
from .verifier import verify_problem
from .rq_score import compute_rq_full, h_prefilter
from .entropy_verl import compute_exact_entropy_from_checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)


def extract_boxed(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def normalize(s: str) -> str:
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


def answers_match(pred: str, gt: str) -> bool:
    if normalize(pred) == normalize(gt):
        return True
    result = _sympy_equal(pred, gt)
    if result is not None:
        return result
    return False


SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)

# ---------------------------------------------------------------------------
# Mutation prompts (used by the SAME model)
# ---------------------------------------------------------------------------

IN_DEPTH_PROMPT = """You are an expert mathematician and Python programmer.

Below is a Python function that generates natural-language math word problems 
using inverse construction (answer chosen first, problem built from it).

```python
{source_code}
```

Modify this function to generate HARDER problems. You may:
- Add more reasoning steps or constraints
- Combine multiple math concepts
- Require multi-step logic (e.g., need intermediate calculations)

RULES:
1. Function MUST be named `generate` and take a single `seed` argument
2. MUST return (problem_text: str, answer: str)
3. problem_text MUST be a natural language word problem (not raw equations)
4. answer MUST be constructed FIRST, then problem built from it
5. Use only standard library + math module
6. Self-contained, no external dependencies

Return ONLY the Python code."""

IN_BREADTH_PROMPT = """You are an expert mathematician and Python programmer.

Below is a Python function that generates math word problems:

```python
{source_code}
```

Create a COMPLETELY DIFFERENT type of math word problem generator covering 
a different topic (e.g., if above does geometry, try probability, 
sequences, work-rate problems, or number puzzles).

RULES:
1. Function MUST be named `generate` and take a single `seed` argument
2. MUST return (problem_text: str, answer: str)  
3. problem_text MUST be a natural language word problem
4. answer MUST be constructed FIRST, then problem built from it
5. Use only standard library + math module
6. Self-contained

Return ONLY the Python code."""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    # Model (SINGLE model for both roles)
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tp: int = 1
    gpu_mem: float = 0.85
    max_tokens: int = 2048
    temperature: float = 0.7

    # Outer loop
    num_epochs: int = 5

    # Inner loop
    num_generations: int = 100
    candidates_per_gen: int = 8
    instances_per_program: int = 3
    in_depth_ratio: float = 0.7

    # H pre-filter
    h_threshold: float = 0.1

    # Rollouts
    num_rollouts: int = 16

    # MAP-Elites
    n_h_bins: int = 6
    n_div_bins: int = 6
    h_range: tuple = (0.0, 5.0)

    # Training
    train_batch_size: int = 256

    # Evaluation (R-Zero benchmarks)
    eval_gsm8k_samples: int = 200    # -1 for full (1319)
    eval_math500_samples: int = 100  # -1 for full (500)
    eval_aime_k: int = 32            # mean@k for AIME

    # Logging
    wandb_project: str = "rq_evolve"

    # Paths
    output_dir: str = "./rq_output"
    seed_programs_dir: str = "./seed_programs"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class EvolutionaryPipeline:
    """
    Self-Evolving pipeline: one model improves at both
    generating problems (Questioner) and solving them (Solver).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_model_path = config.model_path

        self.grid = MAPElitesGrid(
            n_h_bins=config.n_h_bins,
            n_div_bins=config.n_div_bins,
            h_range=config.h_range,
        )

        self._llm = None
        self.evolution_log = []

    # ------------------------------------------------------------------
    # vLLM lifecycle (shared by Questioner + Solver)
    # ------------------------------------------------------------------

    def _load_vllm(self):
        """Load vLLM engine with current model."""
        from vllm import LLM
        logger.info(f"Loading vLLM: {self.current_model_path}")
        self._llm = LLM(
            model=self.current_model_path,
            tensor_parallel_size=self.config.tp,
            gpu_memory_utilization=self.config.gpu_mem,
            trust_remote_code=True,
            max_model_len=4096,
        )

    def _destroy_vllm(self):
        """Destroy vLLM engine and free GPU memory."""
        if self._llm is not None:
            logger.info("Destroying vLLM engine")
            del self._llm
            self._llm = None
            try:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                gc.collect()

    def _build_solver_prompt(self, problem: str) -> str:
        try:
            tok = self._llm.get_tokenizer()
            if hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": problem}],
                    tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            pass
        return f"{SYSTEM_PROMPT}\n\nProblem: {problem}\n\nSolution:"

    def _build_mutator_prompt(self, source: str, mutation_type: str) -> str:
        template = IN_DEPTH_PROMPT if mutation_type == "in_depth" else IN_BREADTH_PROMPT
        raw = template.format(source_code=source)
        try:
            tok = self._llm.get_tokenizer()
            if hasattr(tok, "apply_chat_template"):
                return tok.apply_chat_template(
                    [{"role": "user", "content": raw}],
                    tokenize=False, add_generation_prompt=True,
                )
        except Exception:
            pass
        return raw

    # ------------------------------------------------------------------
    # Questioner: mutate programs (uses same vLLM)
    # ------------------------------------------------------------------

    def _mutate_program(self, parent: ProblemProgram) -> Optional[ProblemProgram]:
        """Use the current model to mutate a problem-generation program."""
        import random as stdlib_random
        from vllm import SamplingParams

        mtype = "in_depth" if stdlib_random.random() < self.config.in_depth_ratio else "in_breadth"
        prompt = self._build_mutator_prompt(parent.source_code, mtype)

        params = SamplingParams(temperature=0.8, max_tokens=2048, n=1)
        outputs = self._llm.generate([prompt], params)
        response = outputs[0].outputs[0].text

        source = self._extract_code(response)
        if source is None or "def generate" not in source:
            return None

        child = ProblemProgram(
            source_code=source,
            parent_id=parent.program_id,
            generation=parent.generation + 1,
            metadata={"mutation_type": mtype},
        )

        # Smoke test
        if child.execute(seed=42, timeout=5.0) is None:
            return None
        return child

    def _extract_code(self, text: str) -> Optional[str]:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        code = m.group(1).strip() if m else text.strip()
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith(("def generate", "import", "from")):
                code = "\n".join(lines[i:])
                break
        return code if "def generate" in code else None

    # ------------------------------------------------------------------
    # Solver: H measurement (Step 5)
    # ------------------------------------------------------------------

    def _measure_h_batch(self, problems: list[str]) -> list[Optional[float]]:
        from vllm import SamplingParams

        prompts = [self._build_solver_prompt(p) for p in problems]
        params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            logprobs=1, n=1,
        )
        outputs = self._llm.generate(prompts, params)

        results = []
        for out in outputs:
            try:
                lps = out.outputs[0].logprobs
                if not lps:
                    results.append(None)
                    continue
                ents = [max(0.0, -next(iter(d.values())).logprob) for d in lps]
                results.append(sum(ents) / len(ents) if ents else None)
            except Exception:
                results.append(None)
        return results

    # ------------------------------------------------------------------
    # Solver: G rollouts (Step 6)
    # ------------------------------------------------------------------

    def _run_rollouts_batch(
        self, instances: list[ProblemInstance]
    ) -> list[list[bool]]:
        from vllm import SamplingParams

        prompts = [self._build_solver_prompt(i.problem) for i in instances]
        params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            n=self.config.num_rollouts,
        )
        outputs = self._llm.generate(prompts, params)

        all_flags = []
        for out, inst in zip(outputs, instances):
            flags = []
            for comp in out.outputs:
                pred = extract_boxed(comp.text)
                flags.append(
                    answers_match(pred, inst.answer) if pred else False
                )
            all_flags.append(flags)
        return all_flags

    # ------------------------------------------------------------------
    # Seed loading + grid init
    # ------------------------------------------------------------------

    def load_seed_programs(self) -> list[ProblemProgram]:
        programs = []
        seed_path = Path(self.config.seed_programs_dir)
        for f in sorted(seed_path.glob("*.py")):
            prog = ProblemProgram(
                source_code=f.read_text(), generation=0,
                metadata={"source_file": f.name},
            )
            inst = prog.execute(seed=42)
            if inst:
                programs.append(prog)
                logger.info(f"Seed OK: {f.name} → {inst.problem[:60]}...")
            else:
                logger.warning(f"Seed FAIL: {f.name}")
        logger.info(f"Loaded {len(programs)} seeds")
        return programs

    def initialize_grid(self, seeds: list[ProblemProgram]):
        for prog in seeds:
            inst = prog.execute(seed=0)
            if inst:
                self.grid.try_insert(
                    program=prog, h_value=1.0,
                    problem_text=inst.problem, rq_score=0.01,
                )
        logger.info(f"Grid init: {self.grid.stats()}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        logger.info("=" * 60)
        logger.info("R_Q Self-Evolving Pipeline")
        logger.info(f"Model: {self.config.model_path}")
        logger.info("=" * 60)

        seeds = self.load_seed_programs()
        if not seeds:
            raise ValueError("No valid seed programs!")
        self.initialize_grid(seeds)

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch+1}/{self.config.num_epochs}")
            logger.info(f"Model: {self.current_model_path}")
            logger.info(f"{'='*60}")

            # --- Phase 1: Evolution (vLLM loaded) ---
            self._load_vllm()

            # Evaluate current model on benchmarks
            eval_results = self._evaluate_benchmarks(epoch)

            self._evolve_questioner(epoch)
            training_data = self._prepare_training_data(epoch)
            self._destroy_vllm()

            # --- Phase 2: GRPO training (veRL subprocess) ---
            ckpt = self._train_solver_verl(training_data, epoch)

            # --- Phase 2.5: 최신 checkpoint로 정확한 엔트로피 정제 ---
            if ckpt:
                self._refine_champion_entropy(ckpt)

            # --- Update model path for next epoch ---
            if ckpt:
                self.current_model_path = ckpt
                logger.info(f"Model updated → {ckpt}")

            # --- Logging ---
            stats = self.grid.stats()
            stats["epoch"] = epoch + 1
            stats["model"] = self.current_model_path
            if eval_results:
                stats["eval"] = {
                    name: {"accuracy": r.accuracy, "metric": r.metric}
                    for name, r in eval_results.items()
                }
            self.evolution_log.append(stats)
            self._save_checkpoint(epoch)
            logger.info(f"Epoch {epoch+1}: {json.dumps(stats, indent=2)}")

        logger.info("\nSelf-Evolving pipeline complete!")
        self._save_final_results()

    # ------------------------------------------------------------------
    # Phase 1: Questioner evolution
    # ------------------------------------------------------------------

    def _evolve_questioner(self, epoch: int):
        logger.info(f"[Phase 1] Evolving ({self.config.num_generations} gens)")

        for gen in range(self.config.num_generations):
            # 1. Sample parents
            parents = [
                p for p in (
                    self.grid.sample_parent()
                    for _ in range(self.config.candidates_per_gen)
                ) if p is not None
            ]
            if not parents:
                continue

            # 2. Mutate (same model as Solver)
            children = []
            for parent in parents:
                child = self._mutate_program(parent)
                if child:
                    children.append(child)
            if not children:
                continue

            # 3-4. Generate + verify
            candidates = []
            for child in children:
                for inst in child.generate_batch(
                    list(range(self.config.instances_per_program))
                ):
                    if verify_problem(inst.problem, inst.answer):
                        inst.verified = True
                        candidates.append((child, inst))
                        break
            if not candidates:
                continue

            # 5. H pre-filter (batched)
            h_vals = self._measure_h_batch(
                [inst.problem for _, inst in candidates]
            )
            surviving = [
                (child, inst, h)
                for (child, inst), h in zip(candidates, h_vals)
                if h is not None and h_prefilter(h, self.config.h_threshold)
            ]
            if not surviving:
                continue

            # 6. G rollouts (batched)
            all_flags = self._run_rollouts_batch(
                [inst for _, inst, _ in surviving]
            )

            # 7. R_Q → grid update
            inserted = 0
            for (child, inst, h), flags in zip(surviving, all_flags):
                rq = compute_rq_full(flags, h)
                child.h_score = h
                child.p_hat = rq.p_hat
                child.rq_score = rq.rq_score
                child.fitness = rq.rq_score
                if self.grid.try_insert(
                    program=child, h_value=h,
                    problem_text=inst.problem, rq_score=rq.rq_score,
                ):
                    inserted += 1

            if (gen + 1) % 10 == 0:
                logger.info(
                    f"  Gen {gen+1}: mut={len(children)} "
                    f"ver={len(candidates)} h_pass={len(surviving)} "
                    f"ins={inserted} cov={self.grid.coverage():.0%} "
                    f"rq={self.grid.mean_rq():.4f}"
                )

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------

    def _prepare_training_data(self, epoch: int) -> list[dict]:
        champions = self.grid.get_all_champions()
        data = []
        base_seed = epoch * 10000
        per_champ = max(
            1, self.config.train_batch_size // max(1, len(champions))
        )
        for champ in champions:
            for i in range(per_champ):
                inst = champ.execute(base_seed + len(data) + i)
                if inst and verify_problem(inst.problem, inst.answer):
                    data.append({
                        "prompt": inst.problem,
                        "answer": inst.answer,
                        "program_id": champ.program_id,
                        "niche": f"{champ.niche_h}_{champ.niche_div}",
                        "rq_score": champ.rq_score,
                    })
        logger.info(f"[Phase 1] {len(data)} problems from {len(champions)} champions")
        return data

    # ------------------------------------------------------------------
    # Phase 2: veRL GRPO training  (+ Phase 2.5 async entropy)
    # ------------------------------------------------------------------

    def _train_solver_verl(
        self, data: list[dict], epoch: int
    ) -> Optional[str]:
        if not data:
            logger.warning("No data, skip training")
            return None

        train_path = self.output_dir / f"train_epoch_{epoch}.parquet"
        self._save_parquet(data, train_path)

        ckpt_dir = self.output_dir / f"verl_ckpt_epoch_{epoch}"

        cmd = [
            "python", "-m", "verl.trainer.main_ppo",
            f"data.train_files={train_path}",
            f"data.train_batch_size={self.config.train_batch_size}",
            "data.max_prompt_length=1024",
            f"data.max_response_length={self.config.max_tokens}",
            f"actor_rollout_ref.model.path={self.current_model_path}",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            f"actor_rollout_ref.rollout.n={self.config.num_rollouts}",
            f"actor_rollout_ref.rollout.temperature={self.config.temperature}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={self.config.tp}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={self.config.gpu_mem}",
            "algorithm.kl_ctrl.kl_coeff=0.001",
            # Custom reward function: extract \boxed{} and compare
            f"custom_reward_function.path={Path('reward_fn.py').resolve()}",
            "custom_reward_function.name=compute_score",
            "trainer.total_epochs=1",
            f"trainer.project_name={self.config.wandb_project}",
            f"trainer.experiment_name=epoch_{epoch}",
            f"trainer.default_local_dir={ckpt_dir}",
        ]

        # Enable wandb if WANDB_MODE is not disabled
        wandb_mode = os.environ.get("WANDB_MODE", "disabled")
        if wandb_mode != "disabled":
            cmd.append("trainer.logger=[console,wandb]")
        else:
            cmd.append("trainer.logger=[console]")

        logger.info(f"[Phase 2] veRL GRPO: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            logger.error(f"veRL failed (code {result.returncode})")
            return None

        for candidate in [
            ckpt_dir / "actor" / "huggingface",
            ckpt_dir / "actor",
            ckpt_dir,
        ]:
            if candidate.exists():
                logger.info(f"Checkpoint → {candidate}")
                return str(candidate)

        return None

    # ------------------------------------------------------------------
    # Phase 2.5: 최신 checkpoint로 정확한 엔트로피 정제
    # ------------------------------------------------------------------

    def _refine_champion_entropy(self, checkpoint_path: str):
        """
        veRL이 저장한 checkpoint(최신 policy)의 HF 모델로
        MAP-Elites 챔피언들의 엔트로피를 재측정한다.

        vLLM(logprobs=1)의 top-1 근사 대신 전체 vocab logits에서
        정확한 Shannon entropy를 계산해 그리드 내 H bin 위치와
        R_Q 점수를 갱신한다.

        순서: Phase 2(veRL) 완료 → 이 메서드 → 다음 Phase 1
        veRL subprocess가 종료된 뒤 실행되므로 GPU가 비어있다.
        """
        champions = self.grid.get_all_champions()
        if not champions:
            return

        problem_texts: list[str] = []
        valid_champions: list[ProblemProgram] = []
        for champ in champions:
            inst = champ.execute(seed=0)
            if inst:
                problem_texts.append(inst.problem)
                valid_champions.append(champ)

        if not problem_texts:
            return

        logger.info(
            f"[Phase 2.5] Exact entropy from latest checkpoint: "
            f"{len(valid_champions)} champions via {checkpoint_path}"
        )

        exact_h_values = compute_exact_entropy_from_checkpoint(
            checkpoint_path=checkpoint_path,
            problems=problem_texts,
            max_new_tokens=256,
            batch_size=4,
        )

        moved = updated = 0
        for champ, new_h, problem in zip(valid_champions, exact_h_values, problem_texts):
            if new_h is None:
                continue
            old_h = champ.h_score
            was_moved = self.grid.rebin_champion(champ, new_h, problem)
            moved += was_moved
            updated += 1
            logger.debug(
                f"  {champ.program_id}: H {old_h:.4f} → {new_h:.4f}"
                f"{' (rebinned)' if was_moved else ''}"
            )

        logger.info(
            f"[Phase 2.5] {updated} updated, {moved} rebinned. "
            f"cov={self.grid.coverage():.0%} mean_rq={self.grid.mean_rq():.4f}"
        )

    def _save_parquet(self, data: list[dict], path: Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            rows = [{
                "data_source": "rq_evolved",
                "prompt": json.dumps([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["prompt"]},
                ], ensure_ascii=False),
                "ability": "math",
                "reward_model": json.dumps({
                    "style": "rule",
                    "ground_truth": r["answer"],
                }, ensure_ascii=False),
                "extra_info": json.dumps({
                    "program_id": r.get("program_id", ""),
                    "niche": r.get("niche", ""),
                    "rq_score": r.get("rq_score", 0.0),
                }, ensure_ascii=False),
            } for r in data]

            table = pa.table({k: [x[k] for x in rows] for k in rows[0]})
            pq.write_table(table, str(path))
            logger.info(f"Saved {len(rows)} rows → {path}")
        except ImportError:
            jsonl = path.with_suffix(".jsonl")
            with open(jsonl, "w") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Benchmark evaluation (R-Zero style)
    # ------------------------------------------------------------------

    def _evaluate_benchmarks(self, epoch: int) -> dict:
        """
        Evaluate current model on standard benchmarks.
        
        Called at the start of each Phase 1 while vLLM is loaded.
        Epoch 0 = baseline; subsequent epochs measure improvement.
        """
        try:
            from evaluation import run_evaluation
            results = run_evaluation(
                llm=self._llm,
                epoch=epoch,
                gsm8k_samples=self.config.eval_gsm8k_samples,
                math500_samples=self.config.eval_math500_samples,
                aime_k=self.config.eval_aime_k,
                max_tokens=self.config.max_tokens,
            )

            # Save detailed results
            eval_dir = self.output_dir / "eval"
            eval_dir.mkdir(exist_ok=True)
            eval_path = eval_dir / f"epoch_{epoch}.json"
            serializable = {}
            for name, res in results.items():
                serializable[name] = {
                    "benchmark": res.benchmark,
                    "accuracy": res.accuracy,
                    "metric": res.metric,
                    "correct": res.correct,
                    "total": res.total,
                    "details": res.details,
                }
            with open(eval_path, "w") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            logger.info(f"[Eval] Results saved → {eval_path}")

            return results

        except Exception as e:
            logger.warning(f"[Eval] Benchmark evaluation failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int):
        d = self.output_dir / f"checkpoint_epoch_{epoch}"
        self.grid.save(str(d / "grid"))
        with open(self.output_dir / "evolution_log.json", "w") as f:
            json.dump(self.evolution_log, f, indent=2)

    def _save_final_results(self):
        summary = {
            "config": {k: v for k, v in vars(self.config).items()
                       if not k.startswith("_")},
            "final_grid_stats": self.grid.stats(),
            "evolution_log": self.evolution_log,
            "final_model": self.current_model_path,
        }
        with open(self.output_dir / "final_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        champions = self.grid.get_all_champions()
        d = self.output_dir / "final_champions"
        d.mkdir(exist_ok=True)
        for c in champions:
            c.save(str(d / f"{c.program_id}.json"))

        logger.info(f"Results → {self.output_dir}")
        logger.info(f"  Final model: {self.current_model_path}")
        logger.info(f"  Champions:   {len(champions)}")
        logger.info(f"  Coverage:    {self.grid.coverage():.0%}")
        logger.info(f"  Mean R_Q:    {self.grid.mean_rq():.4f}")
