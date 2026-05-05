# Uncertainty Metric Report: H vs H_span_max

## 1. 결정 사항

최종 구현과 논문 본문에서는 R_Q의 uncertainty term을 output-token entropy로 통일한다.

```text
R_Q(x) = p_hat(x) * (1 - p_hat(x)) * U(x)
```

여기서 `p_hat(x)`는 rollout 정답률이고, `U(x)`는 아래 두 방식 중 하나이다.

| 이름 | 정의 | 용도 |
|---|---|---|
| `h` | G개 output token entropy의 평균 | 논문/기본 구현 |
| `h_span_max` | response별 reasoning/sentence span-max entropy의 G 평균 | ablation/debug |

`H_input`과 `H_full`은 사용하지 않는다. 현재 veRL actor forward는 prompt+response를 입력으로 받지만, R_Q 계산에 쓰는 `entropies`는 response position에 맞춰 잘린 값이다. 따라서 기본 `h`는 input이나 input+output 전체가 아니라 solver가 생성한 풀이/답변 토큰에 대한 entropy이다.

## 2. H

문제 `x`에 대해 G개 rollout response `y_i`를 생성한다고 하자. response `y_i`의 output token sequence를 `t = 1, ..., T_i`, 각 decoding step의 token distribution을 `p_{i,t}(v)`라고 하면,

```text
H_{i,t} = - sum_{v in V} p_{i,t}(v) log p_{i,t}(v)
H(y_i) = (1 / T_i) * sum_{t=1}^{T_i} H_{i,t}
H(x) = (1 / G) * sum_{i=1}^{G} H(y_i)
```

기본 R_Q는 다음과 같다.

```text
R_Q(x) = p_hat(x) * (1 - p_hat(x)) * H(x)
```

의미는 solver가 output을 생성하는 동안 평균적으로 얼마나 불확실했는지를 보는 것이다. `prop:entropy`의 Fisher/gradient derivation이 직접 정당화하는 대상이 token-level entropy이므로, 본문 기본값은 `h`로 둔다.

## 3. H_span_max

`h_span_max`는 같은 output token entropy `H_t`를 사용하되, 평균을 response 전체에 바로 내지 않고 span 단위로 먼저 묶는다.

명시적 `Step k:` 또는 `Final:` boundary가 있으면 이를 reasoning span으로 사용한다.

```text
S_j = j번째 reasoning span에 속한 output token index 집합
H_span_i(j) = (1 / |S_j|) * sum_{t in S_j} H_{i,t}
H_span_max(y_i) = max_j H_span_i(j)
H_span_max(x) = (1 / G) * sum_{i=1}^{G} H_span_max(y_i)
```

항상 출력이 `Step n:` 형식이라는 보장은 없기 때문에, 현재 구현은 다음 순서로 span을 만든다.

```text
1. Step k: / Final: boundary
2. 문장부호 또는 줄바꿈 기준 sentence/newline span
3. span을 만들 수 없으면 전체 output 평균 H
```

따라서 `h_span_max`의 R_Q는 다음과 같다.

```text
R_Q(x) = p_hat(x) * (1 - p_hat(x)) * H_span_max(x)
```

이 방식은 response 전체 평균에서는 희석되는 특정 풀이 구간의 uncertainty peak를 잡기 위한 변형이다. 다만 한 span의 국소적 불확실성이 문제 전체의 수학적 품질을 보장하지는 않으므로 기본값이 아니라 별도 테스트 지표로 둔다.

## 4. 구현 반영

현재 실제 학습 코드와 feasibility 테스트에서 허용하는 uncertainty metric은 다음 두 개뿐이다.

```text
--uncertainty_metric h
--uncertainty_metric h_span_max
```

설정 파일에서 `*_h.yaml`은 `h`, span ablation 설정은 `h_span_max`를 사용한다. `scripts/test.sh`는 두 지표를 순차 테스트하도록 정리했다.

```sh
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric h --out_dir ./h
python3 scripts/test_feasibility.py --vllm_model /data1/yhoon113/qwen3-4b-base --tp 2 --uncertainty_metric h_span_max --out_dir ./h_span_max
```

## 5. 제외한 방식

이전 실험에서 answer-level uncertainty 방식이 좋은 정량 결과를 보였지만, 해당 방식은 token-level Fisher derivation과 직접 연결되지 않는다. 본문에서 기본값으로 쓰려면 별도 정당화가 필요하므로 최종 구현/논문 본문에서는 제거했다.

최종 결론은 간단하다.

```text
default: h
ablation/debug: h_span_max
```
