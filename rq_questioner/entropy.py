"""
Entropy Measurement: Measures token-level Shannon entropy H_t(x') 
from Solver's forward pass on problem x'.

H(x') = (1/T) * sum_t H_t(x')

This is computed from logits during a single forward pass — no rollout needed.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class EntropyResult:
    """Result of entropy measurement for a single problem."""
    mean_entropy: float          # H_bar = (1/T) * sum H_t
    total_entropy: float         # sum H_t
    num_tokens: int              # T
    token_entropies: list        # [H_1, H_2, ..., H_T]
    problem_text: str = ""


def compute_token_entropy(logits) -> "torch.Tensor":
    """
    Compute Shannon entropy at each token position.
    
    Args:
        logits: (seq_len, vocab_size) tensor of logits
    
    Returns:
        (seq_len,) tensor of per-token entropies in nats
    """
    import torch
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # H_t = -sum p_v log p_v
    return entropy


# @torch.no_grad() -- applied inside function
def measure_entropy_single(
    model,
    tokenizer,
    problem_text: str,
    system_prompt: str = "Solve the following math problem step by step.",
    max_gen_tokens: int = 512,
    device: str = "cuda",
) -> Optional[EntropyResult]:
    """
    Measure entropy of Solver's response to a problem.
    
    Generates a response and computes per-token entropy from the logits.
    This requires a single forward pass (with generation).
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        problem_text: The math problem to measure
        system_prompt: System prompt for the solver
        max_gen_tokens: Maximum tokens to generate
        device: Device to use
    
    Returns:
        EntropyResult with per-token entropies
    """
    try:
        # Format the prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"{system_prompt}\n\nProblem: {problem_text}\n\nSolution:"

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        # Generate with logits
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_gen_tokens,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Extract logits for generated tokens
        if hasattr(outputs, "scores") and outputs.scores:
            all_logits = torch.stack(outputs.scores, dim=0)  # (gen_len, batch, vocab)
            all_logits = all_logits.squeeze(1)  # (gen_len, vocab)

            token_ents = compute_token_entropy(all_logits)
            token_ents_list = token_ents.cpu().tolist()

            T = len(token_ents_list)
            total_h = sum(token_ents_list)
            mean_h = total_h / T if T > 0 else 0.0

            return EntropyResult(
                mean_entropy=mean_h,
                total_entropy=total_h,
                num_tokens=T,
                token_entropies=token_ents_list,
                problem_text=problem_text,
            )
        else:
            return None

    except Exception as e:
        print(f"[Entropy] Error measuring entropy: {e}")
        return None


# @torch.no_grad() -- applied inside function
def measure_entropy_batch(
    model,
    tokenizer,
    problems: list[str],
    **kwargs,
) -> list[Optional[EntropyResult]]:
    """Measure entropy for a batch of problems (sequentially for now)."""
    results = []
    for problem in problems:
        result = measure_entropy_single(model, tokenizer, problem, **kwargs)
        results.append(result)
    return results


def measure_entropy_vllm(
    llm,
    problems: list[str],
    system_prompt: str = "Solve the following math problem step by step.",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[Optional[EntropyResult]]:
    """
    Measure entropy using vLLM for faster inference.
    
    Uses vLLM's SamplingParams with logprobs to extract per-token entropy.
    """
    try:
        from vllm import LLM, SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,  # Get top-1 logprob (we compute entropy from prompt_logprobs)
        )

        # Format prompts
        prompts = []
        for problem in problems:
            prompts.append(f"{system_prompt}\n\nProblem: {problem}\n\nSolution:")

        outputs = llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            try:
                # Extract logprobs from generated tokens
                token_ents = []
                for token_output in output.outputs[0].logprobs:
                    if token_output:
                        # Approximate entropy from available logprobs
                        # With full logprobs this would be exact
                        top_logprob = list(token_output.values())[0].logprob
                        # Rough entropy estimate: -log(p_top) as lower bound
                        ent = -top_logprob
                        token_ents.append(max(0.0, ent))

                T = len(token_ents)
                if T == 0:
                    results.append(None)
                    continue

                total_h = sum(token_ents)
                mean_h = total_h / T

                results.append(EntropyResult(
                    mean_entropy=mean_h,
                    total_entropy=total_h,
                    num_tokens=T,
                    token_entropies=token_ents,
                    problem_text=output.prompt[:200],
                ))
            except Exception:
                results.append(None)

        return results

    except ImportError:
        print("[Entropy] vLLM not available, falling back to sequential measurement")
        return [None] * len(problems)
