"""
Dynamic dataset backed by MAP-Elites grid.

veRL의 RLHFDataset 포맷을 따르면서 MAP-Elites 챔피언 목록을
evolution step 이후 실시간으로 교체할 수 있다.

raw_prompt: SingleTurnAgentLoop이 apply_chat_template 전에 기대하는
            messages 리스트 (dict list, not tokenized)
"""

import threading
import numpy as np
import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


class MapElitesDynamicDataset(Dataset):
    """
    Thread-safe in-memory dataset updated from MAP-Elites champions.

    Each item returns the same schema as RLHFDataset.__getitem__:
      - raw_prompt      : list[dict]  (messages format)
      - dummy_tensor    : Tensor([0], uint8)  (veRL requires non-empty batch)
      - data_source     : str
      - reward_model    : dict   {"ground_truth": answer}
      - extra_info      : dict
      - index           : int
      - tools_kwargs    : dict
      - interaction_kwargs : dict
    """

    def __init__(self, seed_problems: list[dict] | None = None):
        """
        Args:
            seed_problems: list of dicts with keys:
                  problem (str), answer (str), program_id (str), rq_score (float)
        """
        self._lock = threading.Lock()
        self._problems: list[dict] = seed_problems or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, problems: list[dict]):
        """Replace the problem list atomically (called from evolution step)."""
        with self._lock:
            self._problems = list(problems)

    def snapshot(self) -> list[dict]:
        """Return a copy of the current problem list."""
        with self._lock:
            return list(self._problems)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        with self._lock:
            return max(len(self._problems), 1)  # never 0 for DataLoader

    def __getitem__(self, item: int) -> dict:
        with self._lock:
            if not self._problems:
                # Placeholder during cold-start
                problem = "What is 1 + 1?"
                answer = "2"
                program_id = "seed"
                rq_score = 0.0
            else:
                entry = self._problems[item % len(self._problems)]
                problem = entry["problem"]
                answer = entry["answer"]
                program_id = entry.get("program_id", "unknown")
                rq_score = entry.get("rq_score", 0.0)

        raw_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        return {
            "raw_prompt": raw_prompt,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": "rq_evolved",
            "reward_model": {"ground_truth": answer},
            "extra_info": {
                "program_id": program_id,
                "rq_score": rq_score,
            },
            "index": item,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }
