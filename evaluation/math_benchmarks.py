"""
Math benchmark evaluation utilities for in-trainer veRL/vLLM validation.

Aligned with R-Zero's evaluation methodology (datasets_loader.py, generate.py,
results_recheck.py): OpenAI math_500_test CSV for MATH-500, the zwhe99 /
HuggingFaceH4 / yentinglin sources for the other five benchmarks, ×32
inflation for AIME/AMC, the R-Zero system prompt, the boxed-only extraction
regex, and math_verify-based grading.
"""

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from prompts import SOLVER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# R-Zero's evaluation prompt (R-Zero/evaluation/generate.py:29).
MATH_EVAL_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# R-Zero's boxed-only extraction pattern (datasets_loader.py:9).
ANSWER_PATTERN_BOXED = re.compile(r"(?i)\\boxed\s*{([^\n]+)}")

# OpenAI simple-evals CSV for MATH-500 (R-Zero MathDatasetHandler).
MATH500_CSV_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv"
)

# Datasets that R-Zero ×32-inflates for pass@1-with-greedy stability (only
# AMC/AIME suites; MATH/Minerva/Olympiad are not inflated).
INFLATE_X32 = {"amc23", "aime24", "aime25"}


@dataclass
class MathBenchmarkProblem:
    benchmark: str
    problem: str
    answer: str
    metadata: dict[str, Any]
    index: int


def extract_math_answer(text: str) -> tuple[str | None, str]:
    """Return (answer, method) using R-Zero's boxed-only regex (first match)."""
    if not text:
        return None, "none"
    m = ANSWER_PATTERN_BOXED.search(text)
    if m:
        return m.group(1).strip(), "boxed"
    return None, "none"


def _math_verify_equal(pred: str | None, gt: str) -> bool:
    """R-Zero's grader: math_verify.verify(parse(gt), parse(pred))."""
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ImportError(
            "math_verify is required for R-Zero-aligned grading. "
            "Install math-verify==0.7.0 plus latex2sympy2_extended."
        ) from exc
    try:
        return bool(verify(parse(str(gt)), parse(str(pred))))
    except Exception as exc:  # math_verify can raise on weird inputs
        logger.debug(
            "[math_eval] math_verify raised on (gt=%r, pred=%r): %r",
            gt, pred, exc,
        )
        return False


def math_equal(pred: str | None, gt: str) -> bool:
    return _math_verify_equal(pred, gt)


def grade_math_response(response: str, ground_truth: str) -> dict[str, Any]:
    pred, method = extract_math_answer(response)
    return {
        "pred": pred or "",
        "extract_method": method,
        "extracted": pred is not None,
        "boxed": method == "boxed",
        "correct": math_equal(pred, ground_truth),
    }


# ---------------------------------------------------------------------------
# Per-benchmark dataset loaders (R-Zero parity)
# ---------------------------------------------------------------------------

def _math500_cache_path() -> Path:
    base = os.environ.get(
        "EVO_MATH500_CACHE",
        os.path.expanduser("~/.cache/evo-sample/math500"),
    )
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / "math_500_test.csv"


def _load_math500_csv() -> list[dict[str, Any]]:
    """Download (and cache) the OpenAI simple-evals MATH-500 CSV."""
    import pandas as pd

    cache = _math500_cache_path()
    if not cache.exists():
        try:
            import requests

            logger.info("[math_eval] downloading MATH-500 CSV → %s", cache)
            resp = requests.get(MATH500_CSV_URL, timeout=60)
            resp.raise_for_status()
            cache.write_bytes(resp.content)
        except Exception as exc:
            if cache.exists():
                logger.warning(
                    "[math_eval] MATH-500 download failed (%s); using stale cache",
                    exc,
                )
            else:
                raise
    df = pd.read_csv(cache)
    return [row.to_dict() for _, row in df.iterrows()]


def _load_hf(hf_id: str, split: str | None = None, config_name: str | None = None):
    from datasets import load_dataset

    if config_name is not None:
        ds = load_dataset(hf_id, config_name)
        if split is None:
            split = "train"
        return ds[split]
    return load_dataset(hf_id, split=split)


def _load_benchmark_rows(name: str) -> list[dict[str, Any]]:
    """Return raw (question, answer) rows for one benchmark.

    R-Zero ×32-inflates AMC/AIME only; MATH/Minerva/Olympiad stay at their
    natural sizes.
    """
    if name == "math500":
        examples = _load_math500_csv()
        rows = [
            {"question": str(e["Question"]), "answer": str(e["Answer"])}
            for e in examples
        ]
    elif name == "amc23":
        ds = _load_hf("zwhe99/amc23", split="test")
        rows = [{"question": str(r["question"]), "answer": str(r["answer"])} for r in ds]
    elif name == "aime24":
        ds = _load_hf("HuggingFaceH4/aime_2024", split="train")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
    elif name == "aime25":
        ds = _load_hf("yentinglin/aime_2025", split="train", config_name="default")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
    elif name == "minerva_math":
        ds = _load_hf("zwhe99/simplerl-minerva-math", split="test")
        rows = [{"question": str(r["problem"]), "answer": str(r["answer"])} for r in ds]
    elif name == "olympiadbench":
        ds = _load_hf("zwhe99/simplerl-OlympiadBench", split="test")
        rows = []
        for r in ds:
            ans = r["final_answer"]
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            rows.append({"question": str(r["question"]), "answer": str(ans)})
    elif name == "gsm8k":
        ds = _load_hf("openai/gsm8k", split="test", config_name="main")
        rows = []
        for r in ds:
            # GSM8K stores a full worked solution ending in "#### <answer>";
            # keep only the final numeric answer for grading.
            ans = str(r["answer"])
            if "####" in ans:
                ans = ans.split("####")[-1].strip()
            rows.append({"question": str(r["question"]), "answer": ans})
    else:
        raise ValueError(f"unknown benchmark: {name!r}")

    if name in INFLATE_X32:
        rows = list(rows) * 32
    return rows


def load_math_benchmark(
    name: str,
    hf_id: str | None = None,
    split: str | None = None,
    max_samples: int = -1,
    sample_seed: int = 42,
) -> list[MathBenchmarkProblem]:
    """Load one benchmark using R-Zero's source mapping.

    The legacy `hf_id` / `split` arguments are accepted for backward
    compatibility but ignored — sources are determined by `name` to match
    R-Zero's datasets_loader.
    """
    rows = _load_benchmark_rows(name)
    # R-Zero never sub-samples math benchmarks at eval time; honor the
    # `max_samples` knob only for explicit debugging.
    if max_samples is not None and int(max_samples) > 0 and len(rows) > int(max_samples):
        rng = random.Random(sample_seed)
        indices = sorted(rng.sample(range(len(rows)), int(max_samples)))
        rows = [rows[i] for i in indices]

    problems: list[MathBenchmarkProblem] = []
    for idx, item in enumerate(rows):
        question = item.get("question")
        answer = item.get("answer")
        if question is None or answer is None:
            continue
        problems.append(
            MathBenchmarkProblem(
                benchmark=name,
                problem=str(question),
                answer=str(answer),
                metadata={},
                index=idx,
            )
        )
    inflated = " (×32)" if name in INFLATE_X32 else ""
    logger.info("[math_eval] loaded %s: %d examples%s", name, len(problems), inflated)
    return problems


# ---------------------------------------------------------------------------
# Tokenization / dataloader
# ---------------------------------------------------------------------------

class MathBenchmarkDataset(Dataset):
    def __init__(
        self,
        problems: list[MathBenchmarkProblem],
        tokenizer,
        max_prompt_length: int,
        system_prompt: str = MATH_EVAL_SYSTEM_PROMPT,
    ):
        self.problems = list(problems)
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.system_prompt = system_prompt or SOLVER_SYSTEM_PROMPT

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from verl.utils import torch_functional as VF

        item = self.problems[index]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": item.problem},
        ]
        if self.tokenizer.chat_template:
            # R-Zero generate.py:31 — add_special_tokens=True so BOS lands in
            # the prompt string. Pair with add_special_tokens=False at the
            # subsequent encode step to avoid double BOS.
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                add_special_tokens=True,
            )
        else:
            # R-Zero generate.py:33 base-model fallback: system prompt is
            # repeated once at the end of the user turn.
            prompt = (
                f"system: {self.system_prompt}\n"
                f"user: {item.problem}\n"
                f"{self.system_prompt}"
            )

        model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id or 0,
            left_pad=True,
            truncation="left",
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "ground_truth": item.answer,
            "benchmark": item.benchmark,
            "problem": item.problem,
            "problem_index": item.index,
            "metadata": item.metadata,
        }


def build_math_eval_dataloaders(
    math_eval_config,
    tokenizer,
    max_prompt_length: int,
    collate_fn,
) -> dict[str, Any]:
    from torchdata.stateful_dataloader import StatefulDataLoader

    if not getattr(math_eval_config, "enabled", False):
        return {}

    dataloaders = {}
    benchmarks = list(getattr(math_eval_config, "benchmarks", []) or [])
    fast_samples = dict(getattr(math_eval_config, "fast_samples", {}) or {})
    batch_size = int(getattr(math_eval_config, "batch_size", 128) or 128)
    sample_seed = int(getattr(math_eval_config, "sample_seed", 42) or 42)

    for spec in benchmarks:
        if isinstance(spec, str):
            name = spec
        else:
            name = str(spec.get("name"))
        if not name:
            continue
        # R-Zero never sub-samples math benchmarks at eval time. We force
        # full-set evaluation regardless of the legacy fast_samples knob.
        max_samples = int(fast_samples.get(name, -1))
        if max_samples > 0:
            logger.info(
                "[math_eval] %s: ignoring fast_samples=%d (R-Zero alignment requires full set)",
                name,
                max_samples,
            )
        max_samples = -1
        try:
            problems = load_math_benchmark(
                name=name,
                max_samples=max_samples,
                sample_seed=sample_seed,
            )
        except Exception as exc:
            logger.warning("[math_eval] skipping %s: %s", name, exc)
            continue
        if not problems:
            logger.warning("[math_eval] skipping %s: no valid examples", name)
            continue
        dataset = MathBenchmarkDataset(
            problems=problems,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
        )
        dataloaders[name] = StatefulDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
    return dataloaders


def save_math_eval_details(
    output_dir: str | Path,
    global_step: int,
    payload: dict[str, Any],
    outer_iteration: int | None = None,
    epoch: int | None = None,
) -> Path:
    out_dir = Path(output_dir) / "math_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    iteration = outer_iteration if outer_iteration is not None else epoch
    if iteration is None:
        iteration = 0
    prefix = "outer_iteration" if outer_iteration is not None else "epoch"
    path = out_dir / f"math_eval_{prefix}_{iteration}_step_{global_step}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    latest = out_dir / "latest_math_eval.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# GPT-4o re-check (R-Zero results_recheck.py port)
# ---------------------------------------------------------------------------

@dataclass
class GPTJudgeResult:
    """Outcome of a single GPT-4o equivalence check."""
    yes: bool
    raw_response: str | None
    error: str | None


def _gpt_judge_prompt(ground_truth: str, extracted_answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a math answer checker."},
        {
            "role": "user",
            "content": (
                f"Ground truth: {ground_truth}\n"
                f"Model answer: {extracted_answer}\n"
                "Are these mathematically equivalent? "
                "Reply only with 'Yes' or 'No'."
            ),
        },
    ]


def call_gpt_judge(
    ground_truth: str,
    extracted_answer: str,
    *,
    model: str = "gpt-5.4-mini",
    api_key: str | None = None,
    retry_max: int = 3,
    retry_backoff_seconds: list[float] | tuple[float, ...] = (1, 5, 30),
    timeout: float = 30.0,
) -> GPTJudgeResult:
    """Call OpenAI chat completions to grade a single (gt, pred) pair.

    Returns GPTJudgeResult.yes=True iff the model replies 'Yes' (case-insensitive,
    leading token). On all failures, yes=False and `error` is populated; the
    caller should fall back to the math_verify score.
    """
    import time

    if api_key is None:
        return GPTJudgeResult(yes=False, raw_response=None, error="missing_api_key")

    try:
        from openai import OpenAI
    except ImportError as exc:
        return GPTJudgeResult(yes=False, raw_response=None, error=f"openai_import:{exc}")

    client = OpenAI(api_key=api_key, timeout=timeout)
    messages = _gpt_judge_prompt(ground_truth, extracted_answer)
    backoffs = list(retry_backoff_seconds) or [1.0]
    last_error: str | None = None

    for attempt in range(int(retry_max)):
        try:
            if model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
                # Reasoning models reject a custom temperature and the
                # `max_tokens` field; they need `max_completion_tokens` and
                # enough budget for reasoning tokens. `low` effort keeps a
                # yes/no equivalence check fast and cheap.
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=2000,
                    reasoning_effort="low",
                )
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=8,
                )
            text = (resp.choices[0].message.content or "").strip()
            yes = text.lower().startswith("yes")
            return GPTJudgeResult(yes=yes, raw_response=text, error=None)
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if attempt + 1 < int(retry_max):
                sleep_for = backoffs[min(attempt, len(backoffs) - 1)]
                time.sleep(float(sleep_for))
    return GPTJudgeResult(yes=False, raw_response=None, error=last_error or "unknown")
