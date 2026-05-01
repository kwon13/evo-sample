# Uncertainty Metric Ablation: 정량/정성 변화 분석

## 1. 목적

이번 실험의 목적은 R_Q 기반 문제 생성에서 `H_bar`를 어떤 불확실성 지표로 정의할 때 좋은 문제가 더 잘 진화하는지 확인하는 것이다.

기존 R_Q 형태는 동일하게 유지했다.

```text
R_Q = p_hat * (1 - p_hat) * H_bar
```

차이는 `H_bar`를 무엇으로 볼 것인가에 있다. 기존 token-level entropy 계열은 solver의 다음 토큰 분포가 얼마나 퍼져 있는지를 본다. 새로 비교한 지표들은 rollout 결과들이 정답/풀이 단위에서 얼마나 갈라지는지를 더 직접적으로 반영한다.

## 2. 비교한 구현

| 실험 | H_bar 정의 | 의도 | 산출물 상태 |
|---|---|---|---|
| `entropy` | token-level Shannon entropy | 기존 방식. 다음 토큰 분포의 불확실성 측정 | 이번 실행에서는 `dashboard_live.html`만 남아 비교 제외 |
| `gini` | token distribution의 Gini impurity | entropy보다 단순한 분포 불확실성 | 비교 가능 |
| `step_max_entropy` | 풀이 step 중 최대 entropy | 전체 평균이 아니라 가장 헷갈리는 reasoning step 포착 | 비교 가능 |
| `semantic_entropy` | rollout 답변/의미 cluster 분포 entropy | 서로 다른 의미의 풀이/답으로 갈라지는 정도 측정 | 비교 가능 |
| `vote_entropy` | rollout 정답 vote 분포 entropy | 최종 답변들이 얼마나 분산되는지 측정 | 비교 가능 |

핵심 구현 관점에서 보면, `gini`와 `step_max_entropy`는 여전히 token/step 내부의 confidence를 보는 방식이다. 반면 `semantic_entropy`와 `vote_entropy`는 여러 rollout의 결과 분포를 사용하므로, solver가 실제로 어떤 답들 사이에서 흔들리는지를 더 직접적으로 포착한다.

## 3. Entropy 측정 방식과 수식

모든 실험은 최종적으로 동일한 R_Q 공식을 사용한다.

```text
R_Q(x) = p_hat(x) * (1 - p_hat(x)) * U(x)
```

여기서 `p_hat`은 G개 rollout 중 정답률이고, `U(x)`가 실험별 uncertainty score이다. 문서의 `H_bar`는 엄밀히는 선택된 uncertainty metric `U(x)`를 의미한다.

### 3.1 공통 notation

문제 `x`에 대해 solver rollout을 `G`번 수행한다고 하자.

```text
y_i       = i번째 rollout 응답
a_i       = y_i에서 extract한 boxed answer
c_i       = 1[a_i == ground_truth]
p_hat     = (1/G) * sum_i c_i
```

token-level metric에서는 probe rollout 하나의 토큰열을 사용한다.

```text
t = 1, ..., T
V = vocabulary
p_t(v) = t번째 decoding step에서 token v에 대한 solver 확률
```

feasibility script에서는 vLLM top-k logprob만 받기 때문에 `entropy`, `gini`, `step_max_entropy`는 top-k + uniform-tail 근사로 계산했다. 관측된 top-k token 집합을 `K_t`, 나머지 확률질량을 `r_t = 1 - sum_{v in K_t} p_t(v)`라고 하면, tail token에는 `r_t / (|V| - |K_t|)`를 균등하게 배분한다.

### 3.2 `entropy`: token-level Shannon entropy

각 decoding step의 token distribution에 대한 Shannon entropy를 계산하고 평균낸다.

```text
H_t = - sum_{v in V} p_t(v) log p_t(v)
U_entropy(x) = H_bar = (1/T) * sum_t H_t
```

top-k 근사에서는 다음처럼 계산한다.

```text
H_t ~= - sum_{v in K_t} p_t(v) log p_t(v)
       - r_t log( r_t / (|V| - |K_t|) )
```

의미: solver가 다음 토큰 선택에서 얼마나 불확실한지 본다. 장점은 policy 내부 confidence를 직접 측정한다는 점이고, 약점은 애매한 문장이나 낯선 표현도 entropy를 높일 수 있다는 점이다.

### 3.3 `gini`: token-level Gini impurity

Gini impurity는 token distribution이 얼마나 한 token에 몰려 있지 않은지를 본다.

```text
G_t = 1 - sum_{v in V} p_t(v)^2
U_gini(x) = (1/T) * sum_t G_t
```

top-k + uniform-tail 근사는 다음과 같다.

```text
G_t ~= 1 - [ sum_{v in K_t} p_t(v)^2
             + r_t^2 / (|V| - |K_t|) ]
```

의미: Shannon entropy와 비슷하게 token 분포의 퍼짐을 측정하지만, log를 쓰지 않아 스케일이 더 작고 고확률 token에 더 민감하다. 이번 실험에서는 이 스케일 차이 때문에 H bin이 거의 `H0`에만 몰렸다.

### 3.4 `step_max_entropy`: reasoning step 중 최대 entropy

먼저 token-level entropy `H_t`를 구한 뒤, solver 응답에서 `Step k:` 또는 `Final:` 같은 구간을 regex로 찾는다. 각 reasoning step span `S_j`에 속한 token entropy 평균을 구하고, 그중 최댓값을 사용한다.

```text
H_step(j) = (1 / |S_j|) * sum_{t in S_j} H_t
U_step_max(x) = max_j H_step(j)
```

명시적 step boundary가 없으면 fallback으로 전체 평균 entropy를 사용한다.

```text
U_step_max(x) = H_bar
```

의미: 전체 풀이가 평균적으로 불확실하지 않더라도, 특정 reasoning 단계에서 solver가 크게 흔들리는 문제를 포착한다. hard-case mining에는 유리하지만, 한 step의 불확실성이 문제 전체의 수학적 품질을 보장하지는 않는다.

### 3.5 `vote_entropy`: extracted answer vote entropy

G개 rollout에서 추출된 최종 답 `a_i`들의 exact-match class를 만든다. 각 class `k`의 비율을 `q_k`라고 하면, 그 분포의 entropy를 사용한다.

```text
C_k = { i : normalize(a_i) == answer_class_k }
q_k = |C_k| / G
U_vote(x) = - sum_k q_k log q_k
```

만약 추출 답변 문자열이 없으면 fallback으로 정답/오답 binary vote entropy를 쓴다.

```text
q_correct = p_hat
q_wrong   = 1 - p_hat
U_binary = - q_correct log q_correct - q_wrong log q_wrong
```

의미: solver의 최종 답들이 얼마나 여러 후보로 갈라지는지를 본다. token confidence보다 "실제로 다른 답을 내는가"에 더 가깝다. 다만 같은 수학적 답이 `667485`와 `667,485`처럼 표기만 다르면 다른 class로 잡힐 수 있어 answer normalization이 중요하다.

### 3.6 `semantic_entropy`: answer equivalence class entropy

`semantic_entropy`도 rollout answer 분포 entropy를 쓰지만, class를 만들 때 exact string match만 쓰지 않고 SymPy 기반 answer equivalence를 함께 사용한다.

```text
a_i ~ a_j  iff  answer_equiv(a_i, a_j) == True
C_k        = equivalence class under ~
q_k        = |C_k| / G
U_sem(x)   = - sum_k q_k log q_k
```

즉 아래처럼 표기가 달라도 같은 수학적 값이면 같은 class로 묶는 것을 목표로 한다.

```text
1/2, 0.5, 2/4  -> same semantic class
```

의미: solver가 의미적으로 다른 답 사이에서 흔들리는지를 측정한다. `vote_entropy`보다 표기 차이에 덜 민감하지만, SymPy가 처리하지 못하는 형식이나 자연어 답변에는 여전히 취약할 수 있다.

### 3.7 지표 간 핵심 차이

| 방식 | 보는 분포 | rollout 여러 개 사용 | 포착하는 불확실성 | 주요 실패 모드 |
|---|---|---:|---|---|
| `entropy` | token distribution | 아니오 | 다음 토큰 confidence | 애매한 문장에도 높아짐 |
| `gini` | token distribution | 아니오 | token 분포의 impurity | 스케일이 작아 H축 붕괴 가능 |
| `step_max_entropy` | step별 token entropy | 아니오 | 특정 reasoning step의 막힘 | 한 step만 보고 과대평가 가능 |
| `vote_entropy` | extracted answer 분포 | 예 | 최종 답 후보의 분산 | 표기 차이를 다른 답으로 볼 수 있음 |
| `semantic_entropy` | answer equivalence class 분포 | 예 | 의미적으로 다른 답 후보의 분산 | equivalence parser 한계 |

이번 실험에서 `vote_entropy`와 `semantic_entropy`가 더 좋은 정량 결과를 낸 이유는, 이 둘이 token-level 불확실성이 아니라 solver의 최종 행동 분포를 직접 보기 때문이다. R_Q의 목적이 "학습에 유익한 문제"를 찾는 것이라면, 다음 토큰이 흔들리는지보다 여러 풀이/답 사이에서 실제로 갈라지는지가 더 직접적인 신호가 된다.

## 4. 정량 결과

실험은 각 metric별 10 evolution step 결과를 비교했다. 단, `entropy`는 `grid/history/champions/rollout_logs` 파일이 남아 있지 않고 `dashboard_live.html`만 존재한다. HTML 내부의 `HISTORY`, `SNAPSHOTS`, `CHAMPS` 데이터를 파싱하면 `live (step 1/10)` 시점의 중간 결과는 복원할 수 있으므로, 아래 표에는 `entropy`를 partial result로 함께 적었다.

| 방식 | Coverage | Champions | Mean R_Q | Max R_Q | 총 inserted | 해석 |
|---|---:|---:|---:|---:|---:|---|
| `vote_entropy` | **0.500** | **18** | **0.2807** | **0.4195** | **29** | 가장 넓고 안정적으로 archive를 확장 |
| `semantic_entropy` | 0.417 | 15 | 0.2599 | **0.4195** | 16 | 최고점은 같지만 탐색 폭은 vote보다 좁음 |
| `step_max_entropy` | 0.306 | 11 | 0.1887 | 0.3670 | 13 | 특정 hard step 포착은 되지만 확장성은 중간 |
| `entropy` | 0.167 | 6 | 0.1040 | 0.1684 | 1 | HTML 복원값. 단, step 1/10 중간 결과 |
| `gini` | 0.167 | 6 | 0.0604 | 0.1064 | 8 | H축이 거의 열리지 않아 archive가 정체 |

주의: `entropy`는 10-step 완료 결과가 아니므로 `vote_entropy`, `semantic_entropy`, `step_max_entropy`, `gini`와 직접적인 최종 성능 비교는 제한적이다. 다만 중간 결과만 놓고도 H축이 전부 `H0`에 머물렀다는 점은 확인된다.

### 4.1 Coverage 변화

`vote_entropy`는 36개 grid cell 중 18개를 채웠고, `semantic_entropy`는 15개를 채웠다. 반면 `gini`는 6개 cell에 머물렀다. HTML에서 복원한 `entropy`도 step 1 기준 6개 cell만 채웠다. 특히 `gini`와 `entropy`는 모든 champion이 `H0`에만 배치되어 난이도 축이 사실상 열리지 않았다.

```text
vote_entropy      coverage = 0.500
semantic_entropy  coverage = 0.417
step_max_entropy  coverage = 0.306
entropy           coverage = 0.167  (HTML live step 1/10)
gini              coverage = 0.167
```

이는 answer-level uncertainty가 token-level impurity보다 MAP-Elites의 H축을 더 잘 활용했다는 뜻이다.

### 4.2 R_Q 변화

`vote_entropy`와 `semantic_entropy`는 최고 R_Q가 0.4195까지 올라갔다. `step_max_entropy`는 0.3670까지 도달했지만 평균 R_Q가 낮았고, `gini`는 최고점 자체가 0.1064로 크게 낮았다. `entropy`는 step 1 기준 최고 R_Q 0.1684, mean R_Q 0.1040으로 `gini`보다는 높지만, 아직 H축 확장은 보이지 않았다.

`gini`의 낮은 점수는 반드시 문제가 쉬웠다는 의미만은 아니다. 지표 스케일이 entropy 계열보다 작아 H binning과 R_Q 계산에서 불리하게 작동했을 가능성이 크다. 현재 설정에서는 gini를 그대로 쓰면 좋은 후보가 있어도 high-H niche로 올라가기 어렵다.

### 4.3 진화 동역학

`vote_entropy`는 초반부터 많은 cell을 채웠고 후반에도 조금씩 개선됐다.

```text
step 1: coverage 0.417, champions 15, mean R_Q 0.2512
step 5: coverage 0.500, champions 18, mean R_Q 0.2696
step 10: coverage 0.500, champions 18, mean R_Q 0.2807
```

`semantic_entropy`도 초반 성능은 좋았지만 step 5 이후 coverage가 0.417에서 고정됐다.

```text
step 1: coverage 0.361, champions 13, mean R_Q 0.2324
step 5: coverage 0.417, champions 15, mean R_Q 0.2453
step 10: coverage 0.417, champions 15, mean R_Q 0.2599
```

`gini`는 step 2 이후 max R_Q가 더 이상 올라가지 않았고, coverage도 처음부터 끝까지 0.167로 고정됐다.

`entropy`는 HTML에 남은 history가 step 1 하나뿐이다.

```text
step 1: coverage 0.167, champions 6, mean R_Q 0.1040, max R_Q 0.1684
```

따라서 `entropy`에 대해서는 "초기에는 gini보다 높은 R_Q를 보였지만, H0 바깥으로 확장했는지는 확인 불가"로 해석하는 것이 안전하다.

## 5. 정성 결과

정성 분석은 다음 기준으로 보았다.

1. 문제 문장이 자연스러운가
2. 풀이에 필요한 수학적 구조가 실제로 존재하는가
3. 추가 조건이 풀이에 영향을 주는가
4. 서로 다른 도메인이 억지로 붙지 않았는가
5. 정답이 수학적으로 타당한가

### 5.1 좋아진 점

`vote_entropy`와 `semantic_entropy`는 solver가 실제로 답변 수준에서 흔들리는 문제를 잘 찾았다. 그래서 단순히 다음 토큰이 불확실한 문제가 아니라, 여러 풀이 경로 또는 여러 오답 후보가 생기는 문제들이 champion으로 올라왔다.

좋은 예시는 다음과 같다.

```text
8 people each write their name on a card. The cards are shuffled and
redistributed. In how many redistributions do exactly 3 people receive
their own card, and the remaining 5 people are deranged?

answer: 2464
```

이 문제는 고정점 선택과 derangement를 결합해야 하므로 풀이 구조가 명확하다.

```text
Modulo the prime 97, 5 is a primitive root. Let 89 be the least positive
residue of 5^54 modulo 97. How many residue classes x modulo 97 satisfy
x^6 ≡ 89 (mod 97)?

answer: 6
```

이 문제는 primitive root, exponent congruence, modular root counting을 연결하므로 실제 reasoning depth가 있다.

즉 `vote_entropy`/`semantic_entropy` 방식은 "solver가 틀릴 만한 진짜 이유"가 있는 문제를 기존 token confidence 방식보다 더 잘 끌어올렸다.

### 5.2 남은 문제

가장 큰 정성적 약점은 의미 없는 조건 삽입이다.

예를 들어 `vote_entropy`의 상위 champion 중에는 다음과 같은 문장이 섞였다.

```text
the total number of people is a vector of length 8 ...
```

이 조건은 derangement 풀이에 필요하지 않고, 문제 의미도 흐린다.

`gini` 쪽에서는 더 노골적인 장식 조건이 나타났다.

```text
The cards are shuffled and redistributed in a 3D space...
their names are derived from a polynomial with integer coefficients...
```

이런 조건들은 solver uncertainty를 높일 수는 있지만 수학적 난이도를 높이지 않는다. 학습 데이터 관점에서는 reward hacking에 가깝다.

두 번째 문제는 crossover에서 도메인이 억지로 붙는 경우다.

```text
Let a, b, c be positive integers with a + b + c = 8 and gcd(a, b, c) = 1.
What is the maximum value of the product abc, and the remaining 5 people
are deranged?
```

정수 최적화 문제와 derangement 문장이 하나의 수학적 object로 통합되지 않았다. 이런 문제는 R_Q가 높아도 정성적으로는 실패다.

세 번째 문제는 정답 타당성이다.

```text
A 4x4 matrix is constructed such that each row and each column contains
exactly one 1 and the rest are 0s. What is the determinant of this matrix?

answer: 0
```

이 문제는 permutation matrix를 묻고 있는데 determinant는 일반적으로 `±1`이다. 따라서 champion 중에도 수학적으로 틀린 문제가 남아 있다.

`entropy` HTML에서 복원된 상위 champion은 power-of-point secants 문제였다.

```text
From an external point P, two secants intersect a circle at A,B and C,D
respectively, with A and C nearer to P. Given PA=9, AB=21, and PC=5,
find CD.

answer: 49
```

이 문제 자체는 풀이 구조가 명확하다. `PA * PB = PC * PD`를 사용해 `9 * 30 = 5 * PD`, 따라서 `PD = 54`, `CD = 49`가 된다. 다만 HTML의 `source_code`에는 실제 문제에 쓰이지 않는 `determine_parameter()`와 `parameter` 변수가 남아 있어, 생성 프로그램 내부에는 불필요한 장식/잔여 mutation 흔적이 있다.

## 6. 방식별 정성 평가

### `vote_entropy`

정량적으로 가장 좋고, 정성적으로도 가장 쓸 만한 후보를 많이 만든다. 특히 pass rate가 중간이고 rollout 답이 여러 방향으로 갈라지는 문제를 잘 포착한다. 다만 answer vote만 보면 "문장이 이상해서 답이 갈라지는 경우"도 같이 점수를 받을 수 있으므로, 문장 일관성 lint가 필요하다.

### `semantic_entropy`

문제 문장과 풀이 구조가 비교적 자연스럽다. `vote_entropy`보다 coverage는 낮지만, 상위 champion의 품질은 비슷하다. semantic clustering이 답변의 의미 차이를 보므로 단순 표기 차이에 덜 민감한 장점이 있다.

### `step_max_entropy`

solver가 특정 풀이 step에서 막히는 문제를 잘 잡는다. 그러나 한 step의 불확실성이 높다는 이유만으로 전체 문제가 좋은 것은 아니다. 실제로 수학적으로 틀린 matrix determinant 문제가 champion으로 올라왔다. hard-case mining에는 유용하지만 단독 selection metric으로는 위험하다.

### `entropy`

HTML에서 복원된 step 1 결과만 보면 `gini`보다 R_Q 스케일은 높다. 상위 champion도 power-of-point, LCM, derangement, Legendre symbol, AM-GM, committee counting으로 구성되어 있어 초기 seed/archive 품질은 나쁘지 않다. 하지만 모든 champion이 `H0`에 남아 있고, 복원 가능한 history가 1 step뿐이라 최종 탐색 성능은 판단할 수 없다. 정성적으로는 문제 문장 자체는 비교적 자연스럽지만, 상위 power-of-point generator 내부에 쓰이지 않는 parameter logic이 남아 있어 mutation hygiene 문제는 여전히 보인다.

### `gini`

현재 설정에서는 부적합하다. 점수 스케일이 작아 H축이 열리지 않고, champion이 전부 `H0`에 몰렸다. 또한 상위 문제에 의미 없는 장식 조건이 자주 보인다. gini를 쓰려면 metric normalization, H range 재설정, reward-hack lint 강화가 선행되어야 한다.

## 7. 기존 방식 대비 변화 요약

기존 token-level uncertainty 중심 방식은 solver 내부 confidence를 직접 볼 수 있다는 장점이 있다. 하지만 token entropy가 높다는 사실이 반드시 "좋은 수학적 난이도"를 의미하지는 않는다. 애매한 문장, 낯선 표현, 불필요한 조건도 token entropy를 높일 수 있다.

이번 결과에서 `vote_entropy`와 `semantic_entropy`가 더 좋았던 이유는, 이 지표들이 token 분포가 아니라 rollout 결과의 분산을 보기 때문이다. 즉 solver가 실제로 여러 답 또는 여러 의미의 풀이 사이에서 갈라질 때 높은 점수를 준다. 이 차이가 정량적으로는 coverage와 mean R_Q 상승으로 나타났고, 정성적으로는 derangement, modular root counting, CRT counting처럼 풀이 구조가 분명한 hard problem이 더 많이 올라오는 결과로 이어졌다.

다만 answer-level uncertainty도 완전한 해결책은 아니다. 문제 문장이 오염되거나 서로 다른 도메인이 충돌해도 rollout 답이 갈라질 수 있다. 따라서 최종 추천은 다음과 같다.

```text
primary metric: vote_entropy
secondary check: semantic_entropy
auxiliary mining: step_max_entropy
exclude by default: gini
```

## 8. 다음 구현 권장사항

1. `vote_entropy`를 기본 selection metric으로 사용한다.
2. `semantic_entropy`를 tie-breaker 또는 quality check로 같이 기록한다.
3. `step_max_entropy`는 archive selection이 아니라 hard-case 후보 발굴용으로 분리한다.
4. `gini`는 normalization 전까지 selection metric에서 제외한다.
5. mutation lint에 "풀이에 쓰이지 않는 조건"과 "서로 다른 도메인의 문장 충돌"을 추가한다.
6. verifier에 문제별 수학 계약을 더 강화한다. 예를 들어 permutation matrix determinant, AM-GM integer variant, derangement fixed-point count는 deterministic checker를 둘 수 있다.
7. 정답 비교에서 comma normalization을 추가한다. 현재 `667,485`와 `667485`가 다른 답으로 처리되는 사례가 있다.

## 9. 최종 판단

정량적으로는 `vote_entropy`가 가장 우수하다. coverage, champion 수, mean R_Q, inserted 수가 모두 가장 높다. 정성적으로도 좋은 hard problem을 가장 많이 찾지만, 문장 오염과 도메인 충돌을 걸러야 한다.

`semantic_entropy`는 더 보수적인 대안이다. 최고 성능은 `vote_entropy`와 같고 문제 품질도 괜찮지만, 탐색 폭은 더 좁다.

따라서 기존 방식 대신 사용할 구현은 `vote_entropy` 중심의 answer-distribution uncertainty이며, 실제 데이터셋 생성에는 `semantic_entropy` 기반 보조 검증과 rule-based quality lint를 함께 붙이는 구성이 가장 타당하다.
