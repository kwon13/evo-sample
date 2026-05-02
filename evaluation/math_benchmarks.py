"""
Math benchmark evaluation utilities for in-trainer veRL/vLLM validation.

The datasets are expected to use the test-time-compute unified schema:
question, answer, metadata.  The grader follows the Qwen2.5-Math style at a
small scale: extract the final answer from boxed/final-answer patterns, then
use exact normalization and SymPy equivalence.
"""

from __future__ import annotations

import json
import logging
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


MATH_EVAL_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)

_BOXED_RE = re.compile(
    r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:final answer is|final answer:|answer is|answer:)\s*"
    r"(.+?)(?:$|[\n\r])",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+\.\d+|\d+/\d+|\d+)(?:\s*\\?%|%)?"
)


@dataclass
class MathBenchmarkProblem:
    benchmark: str
    problem: str
    answer: str
    metadata: dict[str, Any]
    index: int


def _strip_latex_answer(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.strip("$ \n\t")
    for prefix in ("\\displaystyle", "displaystyle"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s


def extract_math_answer(text: str) -> tuple[str | None, str]:
    """Return (answer, method) from model response text."""
    text = text or ""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return _strip_latex_answer(boxed[-1]), "boxed"

    matches = _FINAL_ANSWER_RE.findall(text)
    if matches:
        candidate = matches[-1].strip().split(".\n")[0].strip()
        return _strip_latex_answer(candidate), "final_phrase"

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return _strip_latex_answer(numbers[-1]), "last_number"
    return None, "none"


def _normalize_answer(s: str) -> str:
    s = _strip_latex_answer(s).lower()
    s = s.rstrip(".")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.replace("\\%", "%")
    for token in ("\\text", "\\mathrm"):
        s = s.replace(token, "")
    for open_ch, close_ch in [("{", "}"), ("[", "]"), ("(", ")")]:
        if s.startswith(open_ch) and s.endswith(close_ch):
            s = s[1:-1]
    return s


def _sympy_equal(a: str, b: str, tol: float = 1e-4) -> bool | None:
    from sympy import N, simplify, sympify
    from sympy.parsing.latex import parse_latex

    def _parse(expr: str):
        expr = _strip_latex_answer(expr)
        expr = expr.replace("^", "**")
        if "\\" in expr:
            try:
                return parse_latex(expr)
            except Exception:
                pass
        try:
            return sympify(expr)
        except Exception:
            return None

    expr_a = _parse(a)
    expr_b = _parse(b)
    if expr_a is None or expr_b is None:
        return None
    try:
        if simplify(expr_a - expr_b) == 0:
            return True
    except Exception:
        pass
    try:
        return abs(float(N(expr_a)) - float(N(expr_b))) <= tol
    except Exception:
        return None


def math_equal(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    pred_norm = _normalize_answer(pred)
    gt_norm = _normalize_answer(gt)
    if pred_norm == gt_norm:
        return True
    result = _sympy_equal(pred, gt)
    if result is not None:
        return result
    return False


def grade_math_response(response: str, ground_truth: str) -> dict[str, Any]:
    pred, method = extract_math_answer(response)
    return {
        "pred": pred or "",
        "extract_method": method,
        "extracted": pred is not None,
        "boxed": method == "boxed",
        "correct": math_equal(pred, ground_truth),
    }


def _load_split(hf_id: str, split: str):
    from datasets import load_dataset

    try:
        return load_dataset(hf_id, split=split)
    except Exception:
        if split != "train":
            logger.warning(
                "[math_eval] failed to load %s split=%s; falling back to train",
                hf_id,
                split,
            )
            return load_dataset(hf_id, split="train")
        raise


def load_math_benchmark(
    name: str,
    hf_id: str,
    split: str = "test",
    max_samples: int = -1,
    sample_seed: int = 42,
) -> list[MathBenchmarkProblem]:
    ds = _load_split(hf_id, split)
    rows = list(ds)
    if max_samples is not None and int(max_samples) > 0 and len(rows) > int(max_samples):
        rng = random.Random(sample_seed)
        indices = sorted(rng.sample(range(len(rows)), int(max_samples)))
        rows = [rows[i] for i in indices]

    problems: list[MathBenchmarkProblem] = []
    for idx, item in enumerate(rows):
        question = item.get("question") or item.get("problem") or item.get("prompt")
        answer = item.get("answer") or item.get("gt") or item.get("ground_truth")
        if question is None or answer is None:
            continue
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"raw_metadata": str(metadata)}
        problems.append(
            MathBenchmarkProblem(
                benchmark=name,
                problem=str(question),
                answer=str(answer),
                metadata=metadata,
                index=idx,
            )
        )
    logger.info("[math_eval] loaded %s: %d examples", name, len(problems))
    return problems


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
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = f"system: {self.system_prompt}\nuser: {item.problem}\nassistant:"

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
            hf_id = f"test-time-compute/{spec}"
            split = "test"
        else:
            name = str(spec.get("name"))
            hf_id = str(spec.get("hf_id"))
            split = str(spec.get("split", "test"))
        if not name or not hf_id:
            continue
        max_samples = int(fast_samples.get(name, -1))
        try:
            problems = load_math_benchmark(
                name=name,
                hf_id=hf_id,
                split=split,
                max_samples=max_samples,
                sample_seed=sample_seed,
            )
        except Exception as exc:
            logger.warning("[math_eval] skipping %s (%s): %s", name, hf_id, exc)
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
    epoch: int,
    global_step: int,
    payload: dict[str, Any],
) -> Path:
    out_dir = Path(output_dir) / "math_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"math_eval_epoch_{epoch}_step_{global_step}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    latest = out_dir / "latest_math_eval.json"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
