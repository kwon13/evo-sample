# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

import json
import random
def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}



def process_image(image: Union[Dict[str, Any], ImageObject, str], min_pixels: int, max_pixels: int) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = ""
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        self.category_performance_json = ""
        raw_perf_json = os.getenv("CATEGORY_PERFORMANCE_JSON", "").strip()
        if raw_perf_json:
            try:
                parsed_perf = json.loads(raw_perf_json)
                self.category_performance_json = json.dumps(parsed_perf, ensure_ascii=False, indent=2)
            except Exception:
                self.category_performance_json = raw_perf_json

        if "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            personas_dataset = load_dataset("proj-persona/PersonaHub", "math", split="train")
            self.personas = [item['input persona'] for item in personas_dataset]
            # self.personas = self.personas.select(range(100))
        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if "questioner_format_with_persona" in self.format_prompt:
            print("load personas")
            return [
                {
                    "role": "system",
                    "content": (
                        f"You are {random.choice(self.personas)}.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if "code_questioner_format" in self.format_prompt:
            if self.category_performance_json:
                performance_json = self.category_performance_json
                system_template = (
                    "You are an expert programming problem setter.\n"
                    "You will be given a JSON summary of model performance by category.\n"
                    "CATEGORY_PERFORMANCE_JSON:\n"
                    "{category_performance_json}\n"
                    "FIRST, in your private scratch-pad, design ONE brand-new competitive-programming problem targeting "
                    "the model's weakest category inferred from CATEGORY_PERFORMANCE_JSON. "
                    "It must require a canonical technique from that category under tight constraints (so brute-force fails), "
                    "and must not resemble famous/textbook problems.\n"
                    "Define a precise stdin/stdout format and constraints.\n"
                    "THEN, still in your private scratch-pad, construct 3-6 non-trivial sample test cases (inputs) that follow "
                    "the spec, cover edge cases, and compute the exact correct outputs.\n"
                    "IMPORTANT: Ensure every sample's stdout is correct. Use ASCII only.\n\n"
                    "FINALLY, without revealing any private thoughts, output **exactly** the following THREE blocks in this order, "
                    "with no extra text, no explanations, and no extra markup:\n\n"
                    "<question>\n"
                    "{Problem statement + Input Format + Output Format + Constraints "
                    "(may include samples in separate block below; do NOT include sample I/O inside <question>)}\n"
                    "</question>\n\n"
                    "<starter_code>\n"
                    "'''{Output ONLY runnable Python starter code: keep the required helper signature but leave its body as pass; "
                    "__main__ must only parse stdin as specified, call the helper, and print.}'''\n"
                    "</starter_code>\n\n"
                    "<samples>\n"
                    "<sample>\n"
                    "<stdin>\n{stdin}\n</stdin>\n"
                    "<stdout>\n{stdout}\n</stdout>\n"
                    "<explanation>\n{explanation}\n</explanation>\n"
                    "</sample>\n"
                    "... repeat 3-6 times ...\n"
                    "</samples>\n"
                )
                system_prompt = system_template.replace("{category_performance_json}", performance_json)
                return [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            "Generate one new, challenging reasoning question now, including starter code and 3-6 samples.\n"
                            "CATEGORY_PERFORMANCE_JSON:\n"
                            f"{performance_json}\n"
                            "Remember to format the output exactly as instructed."
                        )
                    }
                ]
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an expert programming problem setter.\n"
                        "FIRST, in your private scratch-pad, design ONE brand-new competitive-programming problem. "
                        "It must require a canonical technique under tight constraints (so brute-force fails), and must not resemble famous/textbook problems.\n"
                        "Define a precise stdin/stdout format and constraints.\n"
                        "THEN, still in your private scratch-pad, construct 3-6 non-trivial sample test cases (inputs) that follow "
                        "the spec, cover edge cases, and compute the exact correct outputs.\n"
                        "IMPORTANT: Ensure every sample's stdout is correct. Use ASCII only.\n\n"
                        "FINALLY, without revealing any private thoughts, output **exactly** the following THREE blocks in this order, "
                        "with no extra text, no explanations, and no extra markup:\n\n"
                        "<question>\n"
                        "{Problem statement + Input Format + Output Format + Constraints "
                        "(may include samples in separate block below; do NOT include sample I/O inside <question>)}\n"
                        "</question>\n\n"
                        "<starter_code>\n"
                        "'''{Output ONLY runnable Python starter code: keep the required helper signature but leave its body as pass; "
                        "__main__ must only parse stdin as specified, call the helper, and print.}'''\n"
                        "</starter_code>\n\n"
                        "<samples>\n"
                        "<sample>\n"
                        "<stdin>\n{stdin}\n</stdin>\n"
                        "<stdout>\n{stdout}\n</stdout>\n"
                        "<explanation>\n{explanation}\n</explanation>\n"
                        "</sample>\n"
                        "... repeat 3-6 times ...\n"
                        "</samples>\n"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now, including starter code and 3-6 samples.\n"
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if "questioner_format" in self.format_prompt:
            # print('detected questioner_format')
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an expert competition-math problem setter.\n"
                        "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. "
                        "The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. "
                        "Aim for a difficulty such that fewer than 30 % of advanced high-school students could solve it. "
                        "Avoid re-using textbook clichés or famous contest problems.\n"
                        "THEN, without revealing any of your private thoughts, output **exactly** the following two blocks:\n\n"
                        "<question>\n"
                        "{The full problem statement on one or more lines}\n"
                        "</question>\n\n"
                        r"\boxed{final_answer}"
                        "\n\n"
                        "Do NOT output anything else—no explanations, no extra markup."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Generate one new, challenging reasoning question now. "
                        "Remember to format the output exactly as instructed."
                    )
                }
            ]
        if "code_solver_format" in self.format_prompt:
            question = example.get("question", prompt_str)
            starter_code = example.get("starter_code", "")
            pairs = example.get("pairs", [])
            example_block = ""
            if pairs:
                example_block = (
                    "\n\nExample:\n"
                    f"Input:\n{pairs[0][0]}\n\n"
                    f"Output:\n{pairs[0][1]}\n"
                )
            user_content = (
                "Problem:\n"
                f"{question}\n\n"
                "Starter code:\n"
                "```python\n"
                f"{starter_code}\n"
                "```\n"
                f"{example_block}\n"
                "Solve the following problem by completing the given starter code.\n"
            )
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python programmer. Please reason step by step, "
                        "and write a correct Python solution."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
        if "solver_format" in self.format_prompt:
            return [{"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},{"role": "user", "content": prompt_str}]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        if self.tokenizer.chat_template:
            return (
                len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
            )
        else:
            return (
                len("system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]) <= self.max_prompt_length
            )
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            raw_image_data = example.pop(self.image_key)
            images = [
                process_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                for image in raw_image_data
            ]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": raw_image_data}
        else:
            if self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "system: " + messages[0]["content"] + '\n' + "user: " + messages[1]["content"]
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
