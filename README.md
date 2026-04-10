# R_Q Self-Evolving Problem Generation

> **R_Q = p(1-p) . H** 를 fitness로 사용하여, 수학 문제 생성 프로그램을 MAP-Elites로 진화시키고,
> 진화된 문제로 동일 모델을 REINFORCE++로 학습하는 Self-Evolving 프레임워크.

<p align="center">
  <img src="docs/architecture.svg" alt="R_Q Self-Evolving Pipeline Architecture" width="800"/>
</p>

## Overview

LLM의 수학적 추론 능력을 향상시키기 위한 Questioner-Solver 공진화 시스템이다.
기존 접근(R-Zero, AZR)이 RL로 Questioner를 학습시키는 반면,
본 프레임워크는 **문제 텍스트가 아닌 문제 생성 프로그램(Python 함수)을 진화** 시킨다.

**핵심 설계 원리:**

- **역방향 구성(Inverse Construction)**: 정답을 먼저 선택하고 문제를 수학적으로 구성 -> 정답 보장
- **R_Q 목표 함수**: Learnability `p(1-p)`와 gradient strength `H`를 결합한 이론적 근사 (연구 제안서 S2.5)
- **MAP-Elites**: H bins(난이도) x embedding clusters(다양성)으로 다양성 붕괴를 구조적으로 방지
- **Online 학습**: 진화된 문제로 REINFORCE++ 학습 -> Solver 성장 -> R_Q landscape 이동 -> 공진화

## Project Structure

```
evo-sample/
├── run_verl.py                       # veRL 통합 학습 진입점
├── reward_fn.py                      # Solver reward (boxed 추출 + SymPy 비교)
├── configs/
│   ├── rq_config.yaml                # RQ-Evolve 학습 설정
│   └── grpo_config.yaml              # baseline GRPO 설정
├── prompts/                          # 공통 프롬프트 (단일 소스)
│   ├── mutation.py                   # MUTATE_DEPTH, MUTATE_BREADTH, MUTATE_CROSSOVER
│   └── solver.py                     # SOLVER_SYSTEM_PROMPT, SOLVER_COMPLETION_PROMPT
├── seed_programs/                    # 초기 문제 생성 프로그램 (17개)
│   ├── 01_equation_modeling.py       # 다단계 방정식
│   ├── 02_euclidean_geometry.py      # 유클리드 기하
│   ├── 03_gcd_lcm.py                # GCD/LCM
│   ├── 04_remainder_crt.py           # 나머지/CRT
│   ├── 05_combinatorics.py           # 조합론
│   ├── 06_probability.py             # 조건부확률
│   ├── 07_sequence.py                # 수열
│   ├── 08_quadratic.py               # 이차방정식/비에타
│   ├── 09_logarithm.py               # 로그
│   ├── 10_trigonometry.py            # 삼각함수
│   ├── 11_coordinate_geometry.py     # 좌표기하
│   ├── 12_inclusion_exclusion.py     # 포함-배제
│   ├── 13_number_theory.py           # 정수론
│   ├── 14_recurrence.py              # 점화식/재귀
│   ├── 15_linear_algebra.py          # 선형대수
│   ├── 16_calculus.py                # 미적분
│   └── 17_inequalities.py           # 부등식/최적화
├── rq_questioner/                    # 핵심 모듈
│   ├── map_elites.py                 # MAP-Elites grid (embedding D-axis, e-greedy + rank UCB)
│   ├── program.py                    # ProblemProgram (진화 단위)
│   ├── pipeline.py                   # Standalone evolution pipeline
│   ├── verl_trainer.py               # RQEvolveTrainer (RayPPOTrainer + evolution hook)
│   ├── verl_dataset.py               # MapElitesDynamicDataset (thread-safe online update)
│   ├── rq_score.py                   # R_Q = p(1-p) . H
│   ├── verifier.py                   # SymPy 대입 검증
│   └── entropy.py                    # H 측정 유틸리티
├── verl/                             # 내장 veRL 0.3.1 (FSDP + vLLM hybrid engine)
│   ├── trainer/
│   │   ├── ray_trainer.py            # RayPPOTrainer (base, evolution hook 지원)
│   │   ├── core_algos.py             # REINFORCE++, GRPO, GAE 등
│   │   └── config.py                 # PPOConfig
│   └── ...
├── evaluation/                       # 벤치마크 평가 (GSM8K, MATH-500, AIME-2024)
├── scripts/
│   ├── test_local.py                 # CPU 로컬 검증 (GPU 불필요)
│   ├── test_feasibility.py           # GPU evolution feasibility test
│   └── visualize_evolution.py        # 진화 시각화
└── docs/
    └── evolution_pipeline.md         # 진화 파이프라인 상세 문서
```

## Quick Start

### 1. Local Validation (GPU 불필요)

```bash
python scripts/test_local.py
```

Seed 프로그램 실행, R_Q 계산, MAP-Elites grid 동작, pipeline dry run 등 5개 테스트를 검증한다.

### 2. Feasibility Test (GPU 필요)

실제 LLM으로 진화가 동작하는지 검증한다. veRL 없이 vLLM만으로 실행.

```bash
python scripts/test_feasibility.py \
    --vllm_model Qwen/Qwen3-8B-Base \
    --tp 4 \
    --n_evo 50 \
    --n_h_bins 10 \
    --n_div_bins 10
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--vllm_model` | None | vLLM 모델 경로 (미지정 시 mock 모드) |
| `--tp` | 1 | tensor parallel GPU 수 |
| `--n_evo` | 10 | evolution step 수 |
| `--candidates` | 8 | 라운드당 mutation 후보 수 |
| `--max_rounds` | 8 | step당 라운드 수 (고정 budget) |
| `--n_rollouts` | 10 | candidate당 solver rollout 수 (G) |
| `--n_h_bins` | 6 | H축 (entropy/난이도) bin 수 |
| `--n_div_bins` | 6 | D축 (embedding 다양성) bin 수 |
| `--h_range` | 0.0 5.0 | H축 범위 |
| `--h_threshold` | 0.1 | H pre-filter 임계값 |
| `--crossover_ratio` | 0.2 | crossover 연산자 비율 |
| `--in_depth_ratio` | 0.5 | in-depth mutation 비율 |
| `--ucb_c` | 1.0 | UCB exploration 계수 |
| `--epsilon` | 0.3 | e-greedy 탐험 확률 |
| `--temperature` | 0.7 | 생성 온도 |
| `--gpu_mem` | 0.85 | GPU 메모리 사용률 |
| `--out_dir` | ./feasibility_out | 결과 저장 디렉토리 |

결과물:
- `dashboard_live.html`: 실시간 진화 대시보드
- `history_*.json`: step별 coverage, mean_rq, inserted 등
- `champions_*.json`: 최종 champion 프로그램 + 생성 문제 예시
- `grid_*.csv`: 최종 grid 상태 (H x D niche별 R_Q)

### 3. Full Training (veRL + GPU)

MAP-Elites 진화와 REINFORCE++ 학습을 online으로 결합한 전체 학습 파이프라인.

```bash
python run_verl.py --config configs/rq_config.yaml
```

## Algorithm

```
Outer Loop (RayPPOTrainer.fit):
│
│  매 step: dynamic_dataset -> rollout -> reward -> REINFORCE++ update
│
│  evolution_freq마다 (기본: 전체 step의 10%):
│  ┌─────────────────────────────────────────────────────┐
│  │  Inner Loop: Fixed-Budget Evolution (8 rounds x 8)  │
│  │                                                     │
│  │  1. Parent 선택 (e-greedy + rank UCB)                │
│  │  2. LLM mutation (in-depth / in-breadth / crossover)│
│  │  3. Multi-seed 자가 검증 + SymPy 파싱               │
│  │  4. H pre-filter (DPI 부등식 기반)                   │
│  │  5. G rollouts -> p_hat 추정                        │
│  │  6. R_Q = p(1-p) . H_bar 계산                       │
│  │  7. MAP-Elites grid 갱신 (embedding D-axis)         │
│  └─────────────────────────────────────────────────────┘
│
│  _refresh_dataset() -> dataloader 재구성
│  -> 다음 step은 진화된 문제로 학습
│
│  val_freq마다: validation (solver accuracy 추적)
```

## Design Decisions

| 설계 결정 | 선택 | 근거 |
|-----------|------|------|
| 진화 대상 | 문제 생성 프로그램 | FunSearch (2023): 역방향 구성으로 정답 보장 |
| 목표 함수 | R_Q = p(1-p) . H | 연구 제안서 S2.5: gradient update 기여도의 이론적 근사 |
| 다양성 보존 | MAP-Elites grid | Mouret & Clune (2015): niche 독립 진화 |
| D축 | Sentence embedding PCA | 연구 제안서 S3.3: Surface Diversity Illusion 방지 |
| 부모 선택 | e-greedy + rank UCB | Monte Carlo Elites (2023): exploit 편향 제거 |
| 탐색 예산 | 고정 budget | FunSearch: 조기 종료 없이 안정적 탐색 |
| Solver 학습 | REINFORCE++ | KL-regularized RL, GRPO에서 전환 |
| 비용 절감 | H pre-filter -> rollout | 연구 제안서 S4.2: DPI 부등식 기반 계층적 필터링 |

## Feasibility Test Results

Qwen3-8B-Base, 10 steps, 6x6 grid:

```
Step  Coverage  Champions  Inserted  mean_rq  max_rq
  1     31%       11         13      0.244    0.665
  4     42%       15          8      0.323    0.867
  9     50%       18          3      0.373    0.867
```

- Coverage: 31% -> 50% (지속 증가)
- mean_rq: 0.244 -> 0.373 (83% 향상)
- H0~H4까지 5개 난이도 수준, D0~D4까지 5개 의미 클러스터에 분산

## References

- [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) - Romera-Paredes et al., 2023
- [MAP-Elites](https://arxiv.org/abs/1504.04909) - Mouret & Clune, 2015
- [ELM](https://arxiv.org/abs/2206.08896) - Lehman et al., 2022
- [R-Zero](https://arxiv.org/abs/2502.04113) - Yuan et al., 2025
- [Monte Carlo Elites](https://arxiv.org/abs/2104.08781) - Flageat et al., 2023
