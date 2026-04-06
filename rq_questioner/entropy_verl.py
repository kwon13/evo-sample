"""
Exact entropy extraction using a veRL-trained checkpoint.

vLLM은 top-k logprobs만 반환하므로 엔트로피를 근사할 수밖에 없다.
veRL이 checkpoint를 저장한 뒤 HF 모델로 teacher-forcing forward pass를 수행하면
전체 vocab에 대한 logits를 얻어 정확한 Shannon entropy를 계산할 수 있다.

H̄(x') = (1/T) × Σ_t H_t,   H_t = -Σ_{v∈V} p_v log p_v

단계:
  1. Greedy decode로 응답 생성 (결정론적, 짧게)
  2. (prompt + response) 전체를 teacher-forcing으로 forward pass
  3. 응답 위치의 logits에서 정확한 Shannon entropy 계산
  4. GPU 메모리 해제 후 반환
"""

import gc
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


@dataclass
class ExactEntropyResult:
    mean_entropy: float        # H̄ = (1/T) Σ H_t
    token_entropies: list      # [H_1, ..., H_T]
    num_tokens: int            # T (응답 토큰 수)
    problem_text: str = ""


# ---------------------------------------------------------------------------
# 핵심: 단일 forward pass로 정확한 엔트로피 계산
# ---------------------------------------------------------------------------

def _exact_entropy_one(
    model,
    tokenizer,
    problem: str,
    max_new_tokens: int,
    device: str,
) -> Optional[float]:
    """
    하나의 문제에 대해 정확한 Shannon entropy를 계산한다.

    vLLM 방식과의 차이:
      - vLLM(logprobs=k): top-k 확률만으로 H ≈ -log(p_top) 근사
      - 이 함수: 전체 vocab logits → H = -Σ p_v log p_v (정확)
    """
    import torch

    # 1. 프롬프트 구성
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nProblem: {problem}\n\nSolution:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    # 2. Greedy decode (결정론적) — 응답 토큰 확보
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    gen_len = gen_ids.shape[1] - prompt_len
    if gen_len == 0:
        return None

    # 3. Teacher-forcing forward pass → 전체 vocab logits 획득
    #    입력: (prompt + response), 타겟: response 각 위치
    with torch.no_grad():
        output = model(input_ids=gen_ids)
        logits = output.logits  # (1, total_len, vocab_size) — 정확한 분포

    # logit[i]는 token[i+1]을 예측 → 응답 위치: [prompt_len-1 : prompt_len-1+gen_len]
    response_logits = logits[0, prompt_len - 1 : prompt_len - 1 + gen_len, :]

    # 4. 정확한 Shannon entropy: H_t = -Σ_v p_v log p_v
    log_probs = torch.log_softmax(response_logits, dim=-1)   # (gen_len, vocab)
    probs = torch.exp(log_probs)
    token_h = -(probs * log_probs).sum(dim=-1)               # (gen_len,)

    return token_h.mean().item()


# ---------------------------------------------------------------------------
# 배치 처리
# ---------------------------------------------------------------------------

def compute_exact_entropy_from_checkpoint(
    checkpoint_path: str,
    problems: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda",
) -> list[Optional[float]]:
    """
    veRL checkpoint를 HF로 로드해 문제 목록의 정확한 엔트로피를 계산한다.

    Args:
        checkpoint_path: veRL이 저장한 HuggingFace 형식 체크포인트 경로
        problems:        수학 문제 텍스트 목록
        max_new_tokens:  엔트로피 측정용 최대 응답 토큰 수 (짧을수록 빠름)
        batch_size:      한 번에 처리할 문제 수 (GPU OOM 방지)
        device:          "cuda" or "cpu"

    Returns:
        각 문제의 평균 엔트로피 (실패 시 None)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(
        f"[ExactEntropy] Loading checkpoint: {checkpoint_path} "
        f"({len(problems)} problems)"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    results: list[Optional[float]] = []

    try:
        for start in range(0, len(problems), batch_size):
            batch = problems[start : start + batch_size]
            for problem in batch:
                try:
                    h = _exact_entropy_one(
                        model, tokenizer, problem, max_new_tokens, device
                    )
                    results.append(h)
                except Exception as e:
                    logger.warning(f"[ExactEntropy] Failed for problem: {e}")
                    results.append(None)

            logger.debug(
                f"[ExactEntropy] {min(start + batch_size, len(problems))}"
                f"/{len(problems)} done"
            )
    finally:
        # GPU 메모리 반드시 해제 (다음 Phase 1의 vLLM 로드를 위해)
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[ExactEntropy] Model unloaded, GPU memory freed")

    valid = [h for h in results if h is not None]
    if valid:
        logger.info(
            f"[ExactEntropy] Done. mean={np.mean(valid):.4f} "
            f"min={np.min(valid):.4f} max={np.max(valid):.4f} "
            f"({len(valid)}/{len(results)} valid)"
        )

    return results
