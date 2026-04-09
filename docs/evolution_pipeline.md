# R_Q 기반 진화적 문제 생성 파이프라인: 동작 원리

## 1. 전체 구조

이 파이프라인은 **Solver(수학 문제를 푸는 LLM)의 학습 효율을 극대화하는 문제를 자동으로 생성**하는 시스템이다. 핵심 아이디어는 FunSearch(Romera-Paredes et al., 2023)에서 착안한 "문제가 아니라 **문제를 만드는 프로그램**을 진화시킨다"는 접근이다.

```
┌─────────────────────────────────────────────────────────┐
│              Outer Loop: Solver 학습                      │
│                                                          │
│   dynamic_dataset의 문제들로 REINFORCE++ 학습             │
│   (매 step: rollout → reward → advantage → actor update) │
│                                                          │
│   ┌──────────────────────────────────────────┐           │
│   │  evolution_freq마다 Inner Loop 실행       │           │
│   │                                          │           │
│   │  1. 부모 프로그램 선택 (ε-greedy + UCB)   │           │
│   │  2. LLM mutation (코드 변형)              │           │
│   │  3. 자가 검증 (multi-seed + SymPy)        │           │
│   │  4. Solver rollout → p_hat 추정           │           │
│   │  5. Entropy 측정 → H_bar                  │           │
│   │  6. R_Q = p(1-p) · H_bar 계산             │           │
│   │  7. MAP-Elites grid 갱신                  │           │
│   │  8. dataset 교체 → dataloader 재구성      │           │
│   └──────────────────────────────────────────┘           │
│                                                          │
│   → 다음 step은 진화된 문제로 학습                        │
└─────────────────────────────────────────────────────────┘
```


## 2. 왜 R_Q인가: 목표 함수의 이론적 근거

Solver의 학습 목표는 KL-regularized RL objective이다:

```
J(θ) = E_x [ E_y~π_θ [r(x,y)] − β·D_KL(π_θ || π_ref) ]
```

**Proposition 1** (연구 제안서 §2.2): 현재 policy와 최적 policy 사이의 KL divergence를 전개하면, 이진 보상 하에서 `p(1-p)`가 learnability의 이론적 하한이다. `p=0.5`일 때 최대가 되고, `p→0` 또는 `p→1`이면 0으로 붕괴한다.

**Proposition 2** (연구 제안서 §2.4): Policy update의 1차 개선량은 gradient norm의 기댓값에 비례하며, 이는 token-level Shannon entropy `H`와 단조 관계에 있다. LayerNorm 가정 하에서 `C(x') ≍ Σ_t H_t(x')`이다.

이 두 항을 결합하면:

```
R_Q(x') = p_θ(x') · (1 − p_θ(x')) · H_bar(x')
         ├─── learnability ────┤   ├─ gradient strength ─┤
```

이것이 "문제 x'가 Solver의 gradient update에 얼마나 기여하는가"의 이론적 근사이며, **Questioner(문제 생성기)가 최대화해야 할 목표 함수**이다.


## 3. 왜 프로그램을 진화시키는가

기존 접근(Evol-Instruct, WizardLM 등)은 문제 텍스트를 직접 변이시킨다. 하지만 이 방식은 **정답의 정확성을 보장할 수 없다**. LLM이 변형한 문제의 정답이 틀릴 수 있기 때문이다.

본 프레임워크는 **역방향 구성(Inverse Construction)** 원리를 사용하는 Python 프로그램을 진화시킨다:

```python
def generate(seed):
    rng = random.Random(seed)
    # 1단계: 정답을 먼저 선택
    roots = [rng.randint(-8, 8) for _ in range(2)]
    # 2단계: 정답으로부터 문제를 구성
    s, p = sum(roots), roots[0] * roots[1]
    # 3단계: (문제, 정답) 쌍 반환
    problem = f"x^2 + {-s}x + {p} = 0 has two real roots. Find r1^2 + r2^2."
    answer = str(s**2 - 2*p)
    return problem, answer
```

이 구조에서 **정답의 정확성은 프로그램의 수학적 구성에 의해 보장**된다. `(x-a)(x-b)`를 전개한 다항식의 근이 `{a, b}`인 것은 construction에 의한 보장이며, LLM이 판단한 것이 아니다.

이는 FunSearch(Romera-Paredes et al., 2023)의 핵심 조건인 **"easy to evaluate, hard to solve"**를 만족시킨다.


## 4. MAP-Elites: 다양성 보존 전략

### 4.1 왜 MAP-Elites인가

R_Q를 fitness로 사용하는 진화 알고리즘은, fitness만 최대화하면 **높은 R_Q를 주는 좁은 영역으로 수렴**할 위험이 있다. 이것이 다양성 붕괴(Diversity Collapse)이다.

MAP-Elites(Mouret & Clune, 2015)는 행동 공간을 grid로 분할하여, **각 niche가 독립적으로 진화하도록 강제**함으로써 이 붕괴를 구조적으로 방지한다. 빈 niche에 새 프로그램이 배치되면 무조건 champion이 되므로, **미탐색 영역으로의 진화가 자연스럽게 인센티브화**된다.

### 4.2 Grid 축 설계

**H축 (6 bins, 범위 [0.0, 5.0])**: 문제의 난이도/엔트로피 수준.

연구 제안서 §2.6의 DPI 부등식 `p(1-p) ≤ H/2`에 의해, H는 문제의 난이도와 정보이론적으로 묶여 있다. H를 이산화하면 **각 난이도 수준에서 독립적으로 최적 문제가 유지**된다. Solver가 성장하면 특정 H 구간의 champion fitness가 하락하고, 해당 구간에서 새로운 mutation이 champion을 교체한다.

**D축 (10 bins, sentence embedding PCA)**: 문제의 의미적 다양성.

연구 제안서 §3.3의 설계를 따른다. 경량 sentence encoder(all-MiniLM-L6-v2)로 문제 텍스트를 embedding하고, 첫 번째 주성분(PC1)으로 D bin을 결정한다.

```python
# Fitting (seed 문제들로 1회)
embeddings = encoder.encode(seed_problems)  # (N, 384)
_, _, Vt = np.linalg.svd(embeddings - mean)
pc1 = Vt[0]  # 가장 큰 분산 방향

# 새 문제 배정
proj = (encode(new_problem) - mean) @ pc1  # 스칼라 값
div_bin = int((proj - min) / bin_width)
```

이 설계의 장점은 세 가지이다:
1. **외부 대형 모델에 의존하지 않으므로** ablation 공격에 강건하다.
2. **embedding 공간에서의 거리가 의미적 유사성을 반영**하므로, R-Diverse(2024)가 지적한 Surface Diversity Illusion에 강건하다.
3. **연속 공간을 grid로 이산화**하므로, 새로운 유형의 문제가 빈 niche에 자연스럽게 배치된다.

이전 구현에서는 `root_seed_id`(부모의 원본 시드 계열)로 D bin을 결정했는데, 이는 in-breadth mutation으로 완전히 다른 유형의 문제를 만들어도 **부모의 D bin에 강제 배정**되는 문제가 있었다. Embedding 기반으로 전환하여 **실제 문제의 의미에 따라 niche가 결정**되도록 수정했다.


## 5. 부모 선택: ε-greedy + Rank UCB

### 5.1 문제: 순수 UCB의 exploit 편향

원래 MAP-Elites(Mouret & Clune, 2015)는 **uniform random** 부모 선택을 사용한다. 본 구현에서는 UCB1 bandit을 사용하여 exploitation(높은 R_Q 셀 선호)과 exploration(미방문 셀 탐험)을 균형잡으려 했으나, min-max 정규화가 **고R_Q outlier에 의해 왜곡**되어 사실상 같은 셀만 반복 선택하는 문제가 발생했다.

### 5.2 해결: Monte Carlo Elites 방식

Monte Carlo Elites(Flageat et al., 2023)의 접근을 적용한다:

**ε-greedy 혼합**: ε=0.3 확률로 uniform random 선택, (1-ε)=0.7 확률로 UCB 선택.

```python
if random.random() < 0.3:
    return random.choice(occupied_niches).champion  # Exploration
else:
    return argmax(ucb_scores).champion              # Exploitation
```

**Rank 정규화**: min-max 대신 rank 기반 정규화로 outlier에 강건하게 한다.

```python
# 기존 (outlier에 취약):
norm = (rq - min) / (max - min)  # max_rq=1.0인 셀이 항상 score=1.0

# 현재 (rank 기반, 균등 분포):
ranks = argsort(argsort(rqs))   # 순위: 0, 1, 2, ..., N-1
norm = ranks / (N - 1)           # [0, 1] 균등 분포
```


## 6. 고정 Budget 진화 (조기 종료 제거)

### 6.1 문제: 조기 종료로 인한 탐색 부족

이전 구현은 `target_hard_champions=6`에 도달하면 진화를 조기 종료했다. Feasibility test에서 Step 4 이후 이 조건이 항상 충족되어, **8개 candidate만 시도하고 멈추는** 현상이 발생했다.

### 6.2 해결: FunSearch 스타일 고정 예산

FunSearch는 각 island에 **독립적이고 고정된 예산**을 할당한다. 이를 따라:

```python
# 기존: 조기 종료
while True:
    if hard >= target_hard: break      # ← Step 5부터 즉시 종료
    if attempted >= max_attempts: break

# 현재: 고정 8라운드
for round_num in range(1, max_rounds + 1):  # 항상 8라운드 실행
    attempted, inserted = _evolution_round(candidates_per_evo)
```

매 evolution step마다 `8 rounds × 8 candidates = 64 attempts`를 보장한다.


## 7. 진화 라운드의 상세 과정

각 라운드(`_evolution_round`)는 다음 4단계를 거친다:

### Phase 1: Mutation (LLM 코드 변형)

| 연산자 | 비율 | 설명 |
|--------|------|------|
| in-depth | 40% | 같은 유형의 문제를 더 어렵게 (AMC/AIME 수준으로) |
| in-breadth | 40% | 완전히 다른 수학 분야의 문제 생성기로 전환 |
| crossover | 20% | 두 부모 프로그램의 수학적 개념을 결합 |

각 mutation prompt에는 FunSearch 스타일의 **score-aware feedback**이 포함된다:
- 부모 프로그램의 코드
- 현재 p_hat, H, R_Q 점수
- 진단 메시지 ("TOO EASY", "TOO HARD", "Good difficulty")
- Top-3 champion의 코드를 few-shot 예시로 제공

### Phase 2: 자가 검증

생성된 프로그램을 **5개 seed로 실행**하여:
1. 전부 실행 성공하는지 (crash 없음)
2. 모든 answer가 SymPy로 파싱 가능한지 (수학적으로 유효)

실패하면 즉시 폐기. 이 단계의 비용은 ≈0 (순수 코드 실행).

### Phase 3: Solver Rollout + Entropy 측정

**계층적 비용 절감 구조** (연구 제안서 §4.2):

| 단계 | 연산 | 비용 | 탈락률 |
|------|------|------|--------|
| Stage 1 | 프로그램 실행 + 검증 | ≈0 | ~30% |
| Stage 2 | H 측정 (1회 forward pass) | O(T) | ~60% |
| Stage 3 | G=10 rollouts → p_hat 추정 | 10×O(T) | 0% |

H pre-filter(Stage 2)는 DPI 부등식 `p(1-p) ≤ H/2`에 근거한다. **H가 낮으면 R_Q가 구조적으로 낮을 수밖에 없으므로**, 가장 비싼 Stage 3를 호출하기 전에 후보의 약 60%를 제거한다.

### Phase 4: R_Q 계산 + Grid 삽입

```
R_Q = p_hat × (1 - p_hat) × H_bar
```

필터 통과 후 `try_insert()`로 grid에 삽입 시도:
- **빈 niche**: 무조건 삽입 (새 영역 탐험 인센티브)
- **기존 champion 존재**: R_Q가 더 높을 때만 교체 (elitist)


## 8. Dataset 갱신과 Online 학습

진화가 끝나면 `_refresh_dataset()`이 호출된다:

```
모든 niche의 champion 프로그램 수집
    ↓
각 champion에서 16개 seed로 문제 인스턴스 생성
    ↓
dynamic_dataset 교체 + dataloader 재구성
    ↓
다음 학습 step부터 새 문제로 REINFORCE++ 학습
```

이 순환이 **Solver의 능력이 향상되면 → p_hat이 변하고 → R_Q landscape가 이동 → 다음 evolution에서 더 어려운 문제가 선택되는** 공진화 루프를 형성한다.


## 9. 설계 결정의 요약

| 설계 결정 | 선택 | 근거 |
|-----------|------|------|
| 진화 대상 | 문제 생성 **프로그램** | FunSearch: 정답 보장을 위한 역방향 구성 |
| 목표 함수 | R_Q = p(1-p)·H | 연구 제안서 §2.5: gradient update 기여도의 이론적 근사 |
| 다양성 보존 | MAP-Elites grid | Mouret & Clune 2015: niche 독립 진화로 다양성 붕괴 방지 |
| D축 | Sentence embedding PCA | 연구 제안서 §3.3: 의미적 유사성 기반, Surface Diversity Illusion 방지 |
| 부모 선택 | ε-greedy + rank UCB | Monte Carlo Elites: exploit 편향 제거 |
| 탐색 예산 | 고정 budget (8×8=64) | FunSearch: 조기 종료 없이 안정적 탐색 |
| Solver 학습 | REINFORCE++ | 연구 제안서: GRPO에서 RL++로 전환 |
| 비용 절감 | H pre-filter → rollout | 연구 제안서 §4.2: DPI 부등식 기반 계층적 필터링 |
