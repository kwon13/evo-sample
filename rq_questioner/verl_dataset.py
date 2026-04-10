"""
Dynamic dataset backed by MAP-Elites grid.

verl 0.3.1 RLHFDataset 호환: tokenize하여 input_ids 반환.
"""

import threading
import numpy as np
import torch
from torch.utils.data import Dataset
from verl.utils import torch_functional as VF
from prompts import SOLVER_SYSTEM_PROMPT

SYSTEM_PROMPT = SOLVER_SYSTEM_PROMPT


class MapElitesDynamicDataset(Dataset):
    """
    Thread-safe in-memory dataset updated from MAP-Elites champions.
    verl 0.3.1 RLHFDataset과 동일한 __getitem__ 반환 형식.
    """

    def __init__(self, seed_problems=None, tokenizer=None, max_prompt_length=1024):
        self._lock = threading.Lock()
        self._problems = seed_problems or []
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def set_tokenizer(self, tokenizer, max_prompt_length=1024):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def update(self, problems):
        with self._lock:
            self._problems = list(problems)

    def snapshot(self):
        with self._lock:
            return list(self._problems)

    def __len__(self):
        with self._lock:
            return max(len(self._problems), 1)

    def __getitem__(self, item):
        with self._lock:
            if not self._problems:
                problem = "What is 1 + 1?"
                answer = "2"
            else:
                entry = self._problems[item % len(self._problems)]
                problem = entry["problem"]
                answer = entry["answer"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        if self.tokenizer is None:
            # tokenizer 없으면 raw_prompt만 반환 (feasibility test 등)
            return {
                "raw_prompt": messages,
                "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                "data_source": "rq_evolved",
                "reward_model": {"ground_truth": answer},
                "extra_info": {},
                "index": item,
            }

        # tokenize (RLHFDataset과 동일한 형식)
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = f"system: {SYSTEM_PROMPT}\nuser: {problem}"

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
            "ground_truth": answer,
        }
