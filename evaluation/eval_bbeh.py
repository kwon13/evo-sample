"""BBEH (BigBench Extra Hard) evaluation with vLLM.

Supports two modes:
  * Standalone (no --output_dir): appends a single accuracy line to
    `final_results.jsonl` in the current working directory. This matches the
    historical behavior of this script.
  * Per-step (--output_dir DIR): writes `<DIR>/summary.json` and
    `<DIR>/details.jsonl` in the same shape used by `eval_vllm_math.py`, so
    `scripts/run_eval_pipeline.sh` can plug this in alongside the math eval.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import datasets
from transformers import AutoTokenizer


BENCHMARK_NAME = "bbeh"


def extract_last_boxed(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text):
    pattern1 = r'Final Answer:((?:[^<]|<[^<])*?)\n'
    pattern2 = r'The answer is:((?:[^<]|<[^<])*?)\n'
    matches1 = list(re.finditer(pattern1, text))
    matches2 = list(re.finditer(pattern2, text))
    if matches1:
        return matches1[-1].group(1)
    elif matches2:
        return matches2[-1].group(1)
    return None


def extract_solution(solution_str):
    if '<|im_start|>user' in solution_str:
        model_output = re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', solution_str, flags=re.DOTALL, count=1)
    elif 'Assistant:' in solution_str:
        model_output = solution_str.split('Assistant:')[-1].strip()
    else:
        model_output = solution_str

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    extract_boxed_answer = extract_last_boxed(model_output)
    if extract_boxed_answer:
        return extract_boxed_answer
    else:
        return extract_last_final_answer(model_output)


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    if sample is None:
        sample = ""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Extra Hard."""
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


def preprocess_sample(sample: str) -> str:
    if sample is None:
        sample = ""
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory or HF id")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path/id. Defaults to --model_path")
    parser.add_argument("--output_file", type=str, default="outputs.json",
                        help="Standalone mode only: where to dump raw answers (ignored when --output_dir is set).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="If set, write per-step summary.json + details.jsonl here and skip final_results.jsonl.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If >0, randomly sample up to this many examples per category.")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--tensor_parallel_size", "--tp", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=10240)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def _subsample(entries, max_samples: int, seed: int):
    if max_samples is None or max_samples <= 0 or len(entries) <= max_samples:
        return entries
    rng = random.Random(seed)
    return rng.sample(list(entries), max_samples)


def main() -> None:
    from vllm import LLM, SamplingParams

    args = _build_arg_parser().parse_args()
    tokenizer_name = args.tokenizer or args.model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model_path,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    dataset = datasets.load_dataset('MrLight/bbeh-eval')
    categories = sorted(list(set(dataset['train']['task'])))
    print("Categories:", categories)

    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0
    answers = []
    details_rows = []

    print('----------------- Start Answering -------------------')
    start = time.time()

    for category in categories:
        category_entries = [entry for entry in dataset['train'] if entry['task'] == category]
        category_entries = _subsample(category_entries, args.max_samples, args.sample_seed + hash(category) % 10_000)
        prompts = []
        for entry in category_entries:
            query = entry['question'] + '\n'
            messages = [{
                "role": "user",
                "content": query + '\nPlease reason step by step, and put your final answer option within \\boxed{}.'
            }]
            if tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "user: " + query + '\nPlease reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer.'
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

        for entry, output in zip(category_entries, outputs):
            response = output.outputs[0].text
            row = dict(entry)
            row['solution'] = response
            answers.append(row)

            extracted = extract_solution(response)
            correct = evaluate_correctness(extracted, entry['answer'])
            if correct:
                success += 1
                per_category_accuracy[category][0] += 1
            else:
                fail += 1
                per_category_accuracy[category][1] += 1
            details_rows.append({
                "benchmark": BENCHMARK_NAME,
                "category": category,
                "question": entry.get('question'),
                "answer": entry.get('answer'),
                "extracted": extracted,
                "correct": bool(correct),
                "response": response,
            })

        c, w = per_category_accuracy[category]
        total = c + w
        acc = (c / total) if total > 0 else 0.0
        print(f"{category}: {acc:.4f}  ({c}/{total})")

    elapsed = time.time() - start
    total_examples = success + fail
    overall_acc = (success / total_examples) if total_examples > 0 else 0.0

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        details_path = out_dir / "details.jsonl"
        with details_path.open("w", encoding="utf-8") as f:
            for row in details_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        per_category = {}
        for cat, (c, w) in per_category_accuracy.items():
            n = c + w
            per_category[cat] = {
                "correct": int(c),
                "total": int(n),
                "pass_at_1": (c / n) if n > 0 else 0.0,
            }

        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model_path,
            "tokenizer": tokenizer_name,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "n": 1,
                "seed": args.seed,
            },
            "benchmarks": {
                BENCHMARK_NAME: {
                    "pass_at_1": overall_acc,
                    "correct": int(success),
                    "total": int(total_examples),
                    "num_examples": int(total_examples),
                    "elapsed_sec": elapsed,
                    "categories": per_category,
                },
            },
            "overall": {
                "pass_at_1": overall_acc,
                "correct": int(success),
                "total": int(total_examples),
                "num_examples": int(total_examples),
            },
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[{BENCHMARK_NAME}] pass@1={overall_acc*100:.2f}%  n={total_examples}  -> {out_dir}")
    else:
        with open(args.output_file, 'w') as f:
            json.dump(answers, f, indent=2)
        with open('final_results.jsonl', 'a') as f:
            json.dump({"dataset": "bbeh", "model": args.model_path, "accuracy": round(overall_acc * 100, 2)}, f, indent=2)
        print("Overall Accuracy:", overall_acc)


if __name__ == "__main__":
    main()
